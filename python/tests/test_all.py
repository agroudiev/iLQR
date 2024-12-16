import pytest
import ilqr


def test_sum_as_string():
    assert ilqr.sum_as_string(1, 1) == "2"
