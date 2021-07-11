import numpy as np
import pytest
from DataGeneration.GramianAngularField import normalize, polar_encoding, make_gaf


def test_normalize():
    x = np.arange(100)
    x = normalize(x)
    assert (max(x), min(x)) == (1, -1)


def test_polar_encoding():
    x = np.array([1, -1, 0, 0.5])
    time_stamp = np.array([0, 1, 2, 3]) / 4
    assert (polar_encoding(x, time_stamp) == np.array([0, -1. / 4., 0, 1.5 / 4])).any()


@pytest.mark.parametrize("test_input,expected", [(np.array([1]), 1), (np.array([0]), -1)])
def test_make_gaf(test_input, expected):
    assert make_gaf(test_input)[0, 0] == expected
