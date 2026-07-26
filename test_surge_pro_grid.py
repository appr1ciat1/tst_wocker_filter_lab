import json
from pathlib import Path


GRID_PATH = (
    Path(__file__).parent
    / "research"
    / "grids"
    / "surge_pro_layer_ablation_v1.json"
)


def _grid():
    return json.loads(GRID_PATH.read_text(encoding="utf-8"))


def test_surge_pro_grid_is_the_frozen_22_trial_design():
    grid = _grid()
    trials = grid["trials"]
    ids = [trial["trial_id"] for trial in trials]

    assert grid["grid_id"] == "surge-pro-layer-ablation-v1"
    assert len(trials) == grid["acceptance"]["complete_trials"] == 22
    assert len(ids) == len(set(ids))
    assert grid["baseline_trial_id"] in ids
    assert grid["matched_no_sizing_trial_id"] in ids
    assert grid["acceptance"]["cscv_partitions"] == 8
    assert grid["acceptance"]["expected_cscv_splits"] == 70


def test_surge_pro_grid_contains_the_exact_layer_ablation():
    trials = _grid()["trials"]
    neighborhood = [trial for trial in trials if trial["group"] == "tier_neighborhood"]
    no_tiers = [trial for trial in trials if trial["group"] == "no_strong_tiers"]
    no_sizing = [trial for trial in trials if trial["group"] == "no_regime_sizing"]

    assert len(neighborhood) == 18
    assert {
        (
            trial["dynamic_gap_filter"],
            trial["strong_vix_max"],
            trial["max_regime_scale"],
        )
        for trial in neighborhood
    } == {
        (gap, vix, scale)
        for gap in (False, True)
        for vix in (25.0, 28.0, 31.0)
        for scale in (1.7, 1.8, 1.9)
    }
    assert all(trial["regime_sizing"] for trial in neighborhood)
    assert all(trial["strong_tiers"] is not None for trial in neighborhood)

    assert len(no_tiers) == 2
    assert {trial["dynamic_gap_filter"] for trial in no_tiers} == {False, True}
    assert all(trial["regime_sizing"] for trial in no_tiers)
    assert all(trial["strong_tiers"] is None for trial in no_tiers)

    assert len(no_sizing) == 2
    assert {trial["dynamic_gap_filter"] for trial in no_sizing} == {False, True}
    assert all(not trial["regime_sizing"] for trial in no_sizing)
    assert all(trial["strong_tiers"] is None for trial in no_sizing)
