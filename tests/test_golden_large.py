"""Regression at scale: the optimised controller reproduces the unoptimised v2 decisions on 100-1500 tools."""
import json
import os

import pytest

import large_problems as L
from fgao import FGAOConfig, FGAOController

GOLD = json.load(open(os.path.join(os.path.dirname(__file__), "golden_large.json")))


@pytest.mark.parametrize("case", GOLD, ids=lambda g: f"{g['config']}-{g['n']}x{g['m']}")
def test_large_matches_reference(case):
    r = L.run(FGAOController, FGAOConfig, case["n"], case["m"], case["seed"], L.CONFIGS[case["config"]])
    assert r["status"] == case["status"]
    assert [list(t) for t in r["trace"]] == [list(t) for t in case["trace"]]
    assert r["spent"] == case["spent"]
