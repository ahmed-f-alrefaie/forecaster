import pytest


def test_validate_values():
    import forecaster.mr_forecast as mr


    Rmedian, Rplus, Rminus = mr.Mstat2R(mean=1.0, std=0.1, unit='Earth', sample_size=100, classify='Yes')

    assert pytest.approx(Rmedian) == 1.0131364333673698
    assert pytest.approx(Rplus) == 0.10333141892633568
    assert pytest.approx(Rminus) == 0.12829918319291211
