"""MoE load balancing (``model.moe_balance``): the selection-bias update after every step and the sequence-wise
balance loss."""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import Tensor, nn

from prime_rl.configs.sft import MoEBalanceConfig
from prime_rl.trainer.model import iter_moe_blocks
from prime_rl.trainer.models.prime_moe.moe import PrimeMoE, histogram_quantile_biases, sequence_balance
from prime_rl.trainer.parallel_dims import ParallelDims


@torch.no_grad()
def update_selection_biases(routers: list[nn.Module], tokens_per_expert: Tensor, rate: float) -> None:
    """Aux-loss-free load balancing (https://arxiv.org/abs/2408.15664): move each layer's selection bias by ``rate``
    towards underloaded experts, given the step's global ``[num_moe_layers, num_experts]`` counts. The update is
    mean-centred (as in Moonlight and torchtitan), which leaves the selection unchanged but keeps the biases from
    drifting together."""
    for router, load in zip(routers, tokens_per_expert.to(routers[0].selection_bias.device)):
        delta = torch.sign(load.mean() - load)
        router.selection_bias += rate * (delta - delta.mean())


class MoEBalancer:
    """The ``sign`` update works on every MoE layer with a selection bias; the ``quantile`` update and the
    sequence-wise balance loss need ``prime_moe``. Set up before a checkpoint is loaded: the histogram grid is a router
    buffer, so a resumed run continues with the grid it stopped with. ``start_step`` is the first step this run
    trains."""

    def __init__(
        self,
        model: nn.Module,
        config: MoEBalanceConfig,
        micro_batch_size: int,
        max_steps: int | None,
        start_step: int,
        parallel_dims: ParallelDims,
    ):
        self.config = config
        self.moes = list(iter_moe_blocks(model))
        if not self.moes:
            raise ValueError("model.moe_balance needs an MoE model")
        if any(moe.router.selection_bias is None for moe in self.moes):
            raise ValueError("model.moe_balance needs MoE routers with a selection bias")
        if (config.bias_update == "quantile" or config.seq_aux_coeff > 0) and not all(
            isinstance(moe, PrimeMoE) for moe in self.moes
        ):
            raise ValueError("Quantile balancing and the sequence-wise balance loss need model_type 'prime_moe'")
        self.routers = [moe.router for moe in self.moes]
        self.dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
        self.cp_group = parallel_dims.world_mesh["cp"].get_group() if parallel_dims.cp_enabled else None
        self.last_update_step = max_steps - round(config.bias_freeze * max_steps) if config.bias_freeze > 0 else None
        self.histogram = config.bias_update == "quantile" and config.quantile_estimator == "histogram"
        updating = self.last_update_step is None or start_step <= self.last_update_step
        for router in self.routers:
            router.quantile_balance = config.bias_update == "quantile" and updating
            if self.histogram:
                device = router.selection_bias.device
                router.quantile_hist = torch.zeros(router.num_experts, config.quantile_bins, device=device)
                router.quantile_range = torch.tensor([float("inf"), float("-inf")], device=device)
                # The first step's grid; every later step's is the previous step's global range.
                router.register_buffer("quantile_grid", torch.tensor([-1.0, 1.0], device=device))
        self.seq_moes = self.moes if config.seq_aux_coeff > 0 else []
        for moe in self.seq_moes:
            # A micro-batch is `micro_batch_size` packed rows concatenated into one sample; the sequences are the rows.
            moe.balance_rows = micro_batch_size
        self.seq_balance_sum = torch.zeros((), device="cuda")
        self.seq_balance_rows = torch.zeros((), device="cuda")

    def aux_loss_sum(self, row_len: int) -> Tensor | None:
        """The sequence-wise balance loss of the last forward, on the scale of a summed token loss: every one of a
        row's ``row_len`` local tokens carries the row's loss. Under context parallelism the statistics are summed
        over the CP group first, so a row's loss covers the whole row."""
        if not self.seq_moes:
            return None
        loss = torch.zeros((), device="cuda")
        for moe in self.seq_moes:
            counts, score_sums = moe.seq_balance_stats
            full_row_len = row_len
            if self.cp_group is not None:
                dist.all_reduce(counts, group=self.cp_group)
                score_sums = dist_nn.all_reduce(score_sums, group=self.cp_group)
                full_row_len = row_len * dist.get_world_size(self.cp_group)
            balance = sequence_balance(counts, score_sums, full_row_len, moe.router.top_k)
            loss = loss + self.config.seq_aux_coeff * balance.sum() * row_len
            self.seq_balance_sum += balance.detach().sum()
            self.seq_balance_rows += balance.numel()
        return loss

    @torch.no_grad()
    def update(self, step: int, tokens_per_expert: Tensor) -> dict[str, float]:
        """Update the selection biases from step ``step`` (until the freeze) and return the balance metrics."""
        metrics = {}
        if self.seq_moes:
            seq_balance = torch.stack([self.seq_balance_sum, self.seq_balance_rows])
            dist.all_reduce(seq_balance, group=self.dp_cp_group)
            metrics["loss/seq_balance"] = (seq_balance[0] / seq_balance[1]).item()
            self.seq_balance_sum.zero_()
            self.seq_balance_rows.zero_()
        if self.last_update_step is None or step <= self.last_update_step:
            if self.config.bias_update == "sign":
                update_selection_biases(self.routers, tokens_per_expert, self.config.bias_update_rate)
            elif self.histogram:
                metrics |= self.histogram_update()
            else:
                self.micro_batch_mean_update()
        if step == self.last_update_step:
            for router in self.routers:
                router.quantile_balance = False
        biases = torch.stack([router.selection_bias for router in self.routers])
        metrics["router/bias/absmax"] = biases.abs().max().item()
        metrics["router/bias/std"] = biases.std(dim=1).mean().item()
        return metrics

    def micro_batch_mean_update(self) -> None:
        """Set each layer's selection bias to minus the step's balancing thresholds of its experts
        (``PrimeRouter.record_quantiles``), averaged over the micro-batches and ranks and mean-centred."""
        sums = torch.stack([torch.cat([router.quantile_sum, router.quantile_count[None]]) for router in self.routers])
        dist.all_reduce(sums, group=self.dp_cp_group)
        for router, row in zip(self.routers, sums):
            thresholds = row[:-1] / row[-1]
            router.selection_bias.copy_(thresholds.mean() - thresholds)
            router.quantile_sum.zero_()
            router.quantile_count.zero_()

    def histogram_update(self) -> dict[str, float]:
        """Sum the step's histograms over the ranks, set each layer's selection bias from them
        (``histogram_quantile_biases``) and use the step's global range of the required biases as the next step's
        grid. Returns the grid's mean bin width and the share of values in its edge bins, which also hold the values
        outside it."""
        hists = torch.stack([router.quantile_hist for router in self.routers])
        dist.all_reduce(hists, group=self.dp_cp_group)
        ranges = torch.stack(
            [torch.stack([-router.quantile_range[0], router.quantile_range[1]]) for router in self.routers]
        )
        dist.all_reduce(ranges, op=dist.ReduceOp.MAX, group=self.dp_cp_group)
        bin_width = edge_share = 0.0
        for router, hist, (neg_lo, hi) in zip(self.routers, hists, ranges):
            router.selection_bias.copy_(histogram_quantile_biases(hist, router.quantile_grid, router.top_k))
            bin_width += ((router.quantile_grid[1] - router.quantile_grid[0]) / hist.shape[1]).item()
            edge_share += ((hist[:, 0].sum() + hist[:, -1].sum()) / hist.sum()).item()
            router.quantile_grid.copy_(torch.stack([-neg_lo, torch.maximum(hi, -neg_lo + 1e-6)]))
            router.quantile_hist.zero_()
            router.quantile_range.copy_(torch.tensor([float("inf"), float("-inf")]))
        return {
            "router/qb/bin_width": bin_width / len(self.routers),
            "router/qb/edge_share": edge_share / len(self.routers),
        }
