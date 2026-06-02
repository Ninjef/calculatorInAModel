import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "diagnose_amortized_prior_from_trace.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnose_amortized_prior_from_trace", SCRIPT_PATH
)
prior_diag = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = prior_diag
SPEC.loader.exec_module(prior_diag)


def test_parse_int_set_accepts_comma_separated_values():
    assert prior_diag.parse_int_set("1, 3,5") == {1, 3, 5}


def test_parse_int_set_allows_empty_default():
    assert prior_diag.parse_int_set("") == set()


def test_row_route_reads_calculator_hook_route():
    assert prior_diag.row_route({"calculator_hook_route": "2"}) == 2


def test_row_route_requires_calculator_hook_route():
    with pytest.raises(ValueError):
        prior_diag.row_route({})
