import torch

from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.prime_moe.moe import (
    PrimeRouter,
    histogram_quantile_biases,
    sequence_balance,
    sequence_balance_stats,
)
from prime_rl.trainer.moe_balance import update_selection_biases


def test_update_selection_biases_moves_towards_underloaded_experts():
    router = MoE.from_args(MoEArgs(num_experts=4), dim=4, hidden_dim=8, shared_expert=None).router
    update_selection_biases([router], torch.tensor([[9.0, 1.0, 1.0, 1.0]]), rate=0.1)
    update_selection_biases([router], torch.tensor([[9.0, 1.0, 1.0, 1.0]]), rate=0.1)
    # sign(mean - load) = [-1, 1, 1, 1], centred to [-1.5, 0.5, 0.5, 0.5] per step
    torch.testing.assert_close(router.selection_bias, torch.tensor([-0.3, 0.1, 0.1, 0.1]))


def test_sequence_balance_of_a_row_split_over_cp_ranks_is_the_whole_rows():
    scores = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
    selected = torch.tensor([[0], [0], [0], [1]])
    # row 0 sends both tokens to expert 0: f = [2, 0], P = [0.9, 0.1]; row 1 is balanced: f = [1, 1], P = [0.5, 0.5]
    counts, score_sums = sequence_balance_stats(scores, selected, num_rows=2)
    torch.testing.assert_close(sequence_balance(counts, score_sums, row_len=2, top_k=1), torch.tensor([1.8, 1.0]))
    # One row of 4 tokens held by two CP ranks: their summed statistics give the whole row's balance.
    whole = sequence_balance(*sequence_balance_stats(scores, selected, num_rows=1), row_len=4, top_k=1)
    halves = [sequence_balance_stats(scores[i : i + 2], selected[i : i + 2], num_rows=1) for i in (0, 2)]
    summed = [sum(stats) for stats in zip(*halves)]
    torch.testing.assert_close(sequence_balance(*summed, row_len=4, top_k=1), whole)


def test_histogram_quantile_balancing_balances_the_recorded_batch_however_it_is_split():
    torch.manual_seed(0)
    router = PrimeRouter(
        dim=16, num_experts=6, num_null_experts=2, top_k=2, route_norm=True, route_scale=1.0, bias_on_logits=True
    )
    with torch.no_grad():
        router.gate.weight[0] += 0.5  # expert 0 is preferred
    router.quantile_balance = True
    router.register_buffer("quantile_grid", torch.tensor([-4.0, 4.0]))
    x = torch.randn(400, 16)
    hists = []
    for chunks in (1, 4):
        router.quantile_hist = torch.zeros(8, 1024)
        router.quantile_range = torch.tensor([float("inf"), float("-inf")])
        for part in x.chunk(chunks):
            router(part)
        hists.append(router.quantile_hist)
    torch.testing.assert_close(hists[0], hists[1])

    # One update brings every expert, null experts included, close to 400 * 2 / 8 = 100 tokens.
    router.selection_bias.copy_(histogram_quantile_biases(hists[0], router.quantile_grid, router.top_k))
    with torch.no_grad():
        counts = router(x)[2]
    assert counts.max() / counts.float().mean() < 1.15
