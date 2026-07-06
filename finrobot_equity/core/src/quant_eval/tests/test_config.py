"""Tests for EvalConfig (quant_eval/config.py)."""
from quant_eval.config import EvalConfig


def test_monthly_grid_is_twelve_periods():
    assert EvalConfig(start="2023-01-01", end="2024-12-31").periods_per_year == 12


def test_frequency_maps_to_periods():
    assert EvalConfig(start="a", end="b", grid_freq="W").periods_per_year == 52
    assert EvalConfig(start="a", end="b", grid_freq="QE").periods_per_year == 4
    assert EvalConfig(start="a", end="b", grid_freq="2ME").periods_per_year == 6


def test_unknown_freq_defaults_to_twelve():
    assert EvalConfig(start="a", end="b", grid_freq="???").periods_per_year == 12


def test_roundtrip_serialization():
    cfg = EvalConfig(start="2023-01-01", end="2024-12-31", round_trip_bps=15.0, seed=7)
    assert EvalConfig.from_dict(cfg.to_dict()) == cfg


def test_from_dict_ignores_unknown_keys():
    cfg = EvalConfig.from_dict({"start": "a", "end": "b", "not_a_field": 1})
    assert cfg.start == "a" and cfg.end == "b"
