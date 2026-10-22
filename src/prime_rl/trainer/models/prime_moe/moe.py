from typing import Literal

import torch
from torch import nn

from prime_rl.trainer.distributed.token_dispatcher import TokenDispatcher
from prime_rl.trainer.models.layers.activations import Activation
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import (
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
    record_moe_routing_statistics,
)
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig


class SituGlu(Activation):
    """Sigmoid Tanh Unit GLU (Kimi K3 report, eq. 12): SwiGLU with both factors soft-capped,
    ``(4 tanh(gate / 4) * sigmoid(gate)) * (25 tanh(up / 25))``; close to SwiGLU near 0 and bounded by 100."""

    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        assert gate is not None
        return 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * (25.0 * torch.tanh(up / 25.0))


class PrimeRouter(TokenChoiceTopKRouter):
    """Sigmoid-scored top-k router with a selection bias and ``num_null_experts`` zero-output experts next to the
    routed ones (AdaMoE). Null experts are scored, selected and balanced like the others, so a token that selects
    them uses fewer routed experts."""

    def __init__(
        self,
        *,
        dim: int,
        num_experts: int,
        num_null_experts: int,
        top_k: int,
        route_norm: bool,
        route_scale: float,
    ):
        super().__init__(
            dim=dim,
            num_experts=num_experts + num_null_experts,
            top_k=top_k,
            score_func="sigmoid",
            route_norm=route_norm,
            route_scale=route_scale,
            selection_bias=True,
        )
        self.num_null_experts = num_null_experts


class PrimeMoE(MoE):
    """The shared token-choice MoE runtime with a ``PrimeRouter``. With null experts, tokens use different numbers of
    routed experts: only the (token, routed expert) pairs are dispatched, one per row, and ``routed_k_counts`` counts
    the training tokens per number of routed experts used.

    With ``latent_dim`` (LatentMoE), the routed experts work in that width: the token is projected down before the
    dispatch and the combined expert output is projected back up, with an RMS norm on the expert input
    (``latent_norm = "input"``, Marin) or on the combined output before the up-projection (``"output"``, Kimi K3's
    Stable LatentMoE). The router and the shared experts read the full-width token."""

    def __init__(
        self,
        *,
        router: PrimeRouter,
        experts: GroupedExperts,
        shared_expert: FeedForward | None,
        dim: int,
        latent_dim: int | None,
        latent_norm: Literal["input", "output"],
        norm_eps: float,
    ) -> None:
        super().__init__(
            router=router,
            experts=experts,
            shared_expert=shared_expert,
            score_before_experts=False,
            load_balance_coeff=1e-3,
        )
        self.variable_k = router.num_null_experts > 0
        self.set_token_dispatcher(self.token_dispatcher)
        # Usage of every expert, null experts included.
        self.register_buffer(
            "tokens_per_expert", torch.zeros(router.num_experts, dtype=torch.float32), persistent=False
        )
        # Training tokens per number of routed experts used, for variable-k routing.
        self.register_buffer(
            "routed_k_counts",
            torch.zeros(router.top_k + 1, dtype=torch.float32) if self.variable_k else None,
            persistent=False,
        )
        self.latent_down = self.latent_norm = self.latent_up = None
        self.norm_latent_output = latent_norm == "output"
        if latent_dim is not None:
            self.latent_down = nn.Linear(dim, latent_dim, bias=False)
            self.latent_norm = RMSNorm(RMSNormConfig(hidden_size=latent_dim, eps=norm_eps))
            self.latent_up = nn.Linear(latent_dim, dim, bias=False)

    def prepare_expert_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_down is None:
            return x
        x = self.latent_down(x)
        return x if self.norm_latent_output else self.latent_norm(x)

    def prepare_expert_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_up is None:
            return x
        if self.norm_latent_output:
            x = self.latent_norm(x)
        return self.latent_up(x)

    def set_token_dispatcher(self, token_dispatcher: TokenDispatcher) -> None:
        # Variable-k routing dispatches one (token, expert) pair per row.
        token_dispatcher.top_k = 1 if self.variable_k else self.router.top_k
        self.token_dispatcher = token_dispatcher

    def forward(self, x: torch.Tensor, routed_experts: torch.Tensor | None = None) -> torch.Tensor:
        assert routed_experts is None, "PrimeMoE does not support router replay"
        bs, slen, dim = x.shape
        x = x.view(-1, dim)
        top_scores, selected_experts_indices, num_tokens_per_expert, routing_confidence_sum = self.router(x)

        with torch.no_grad():
            record_moe_routing_statistics(
                self.tokens_per_expert, self.routing_confidence_sum, num_tokens_per_expert, routing_confidence_sum
            )

        expert_input = self.prepare_expert_input(x)
        if self.variable_k:
            # Dispatch only the (token, routed expert) pairs, without the null experts.
            routed = selected_experts_indices < self.experts.num_experts
            if torch.is_grad_enabled():
                with torch.no_grad():
                    # scatter_add_ instead of bincount: CUDA bincount reads its input's maximum to the host.
                    k = routed.sum(dim=1)
                    self.routed_k_counts.scatter_add_(0, k, torch.ones_like(k, dtype=self.routed_k_counts.dtype))
            token_indices, slots = routed.nonzero(as_tuple=True)
            pair_output = self.token_dispatcher.run(
                expert_input[token_indices],
                top_scores[token_indices, slots].unsqueeze(1),
                selected_experts_indices[token_indices, slots].unsqueeze(1),
                self.experts,
                score_before_experts=self.score_before_experts,
            )
            routed_output = (
                torch.zeros(expert_input.shape, dtype=torch.float32, device=x.device)
                .index_add(0, token_indices, pair_output.float())
                .to(x.dtype)
            )
        else:
            routed_output = self.token_dispatcher.run(
                expert_input,
                top_scores,
                selected_experts_indices,
                self.experts,
                score_before_experts=self.score_before_experts,
            )

        shared_output = self.shared_expert(x) if self.shared_expert is not None else None
        self.token_dispatcher.synchronize()
        routed_output = self.prepare_expert_output(routed_output)
        if shared_output is not None:
            routed_output = routed_output + shared_output
        return routed_output.reshape(bs, slen, dim)

    def init_weights(self, init_std: float, buffer_device: torch.device):
        super().init_weights(init_std, buffer_device)
        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(self.router.num_experts, dtype=torch.float32)
            self.router.selection_bias = torch.zeros(self.router.num_experts, dtype=torch.float32)
            if self.routed_k_counts is not None:
                self.routed_k_counts = torch.zeros_like(self.routed_k_counts)
