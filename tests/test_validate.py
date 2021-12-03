import hypothesis
import pytest
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from forecaster.mr_forecast import load_file


all_hyper = load_file()

def test_validate_values():
    import forecaster.mr_forecast as mr


    Rmedian, Rplus, Rminus = mr.Mstat2R(mean=1.0, std=0.1, unit='Earth', sample_size=100)
    assert pytest.approx(Rmedian, rel=1e-1) == 1.0025252064742383
    # assert pytest.approx(Rplus, rel=1) == 0.11824078730136178
    # assert pytest.approx(Rminus, rel=1e-1) == 0.10105473910186069



@given(mean=st.floats(1e-3, 1e3, allow_nan=False, allow_infinity=False),
            std=st.floats(1e-4, 1, allow_nan=False, allow_infinity=False))
def test_validate_linear_piece(mean, std):
    from forecaster.func import generate_mass, pick_random_hyper, \
        piece_linear, piece_linear_II

    mass = generate_mass(mean, std, 10)
    sample_size = len(mass)
    logm = np.log10(mass)
    prob = np.random.random(sample_size)
    logr = np.ones_like(logm)

    hyper = pick_random_hyper(all_hyper, sample_size=sample_size)

    result = piece_linear_II(hyper, logm, prob)

    expected = np.array([piece_linear(hyper[i], logm[i], prob[i]) 
                         for i in range(sample_size)])

    np.testing.assert_array_equal(result, expected)

