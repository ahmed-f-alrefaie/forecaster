import numpy as np
import pytest
from forecaster.mr_forecast import load_file

NSAMPLES = 100

@pytest.mark.linearbench
def test_piece_linear_original(benchmark):
    from forecaster.func import generate_mass, pick_random_hyper, \
        piece_linear, piece_linear_II
    from forecaster.mr_forecast import load_file

    all_hyper = load_file()
    
    nsamples = NSAMPLES
    mass = generate_mass(1.0, 0.1, nsamples)
    sample_size = len(mass)
    logm = np.log10(mass)
    prob = np.random.random(sample_size)
    logr = np.ones_like(logm)
    hyper = pick_random_hyper(all_hyper, sample_size=sample_size)
    def myfunc():
        return [piece_linear(hyper[i], logm[i], prob[i]) for i in range(sample_size)]

    benchmark(myfunc)

@pytest.mark.linearbench
def test_piece_linear_new(benchmark):
    from forecaster.func import generate_mass, pick_random_hyper, \
        piece_linear, piece_linear_II
    from forecaster.mr_forecast import load_file

    all_hyper = load_file()
    
    nsamples = NSAMPLES
    mass = generate_mass(1.0, 0.1, nsamples)
    sample_size = len(mass)
    logm = np.log10(mass)
    prob = np.random.random(sample_size)
    logr = np.ones_like(logm)
    hyper = pick_random_hyper(all_hyper, sample_size=sample_size)

    benchmark(piece_linear_II, hyper, logm, prob)

@pytest.mark.probrbench
def test_probR_original(benchmark):
    from scipy.stats import norm, truncnorm
    from forecaster.func import generate_mass, pick_random_hyper, \
        piece_linear, piece_linear_II, ProbRGivenM

    all_hyper = load_file()
    mean = 0.01
    std = 0.001
    sample_size = NSAMPLES
    radius = truncnorm.rvs( (0.-mean)/std, np.inf, loc=mean, scale=std, size=sample_size)
    logr = np.log10(radius)
    logm = np.ones_like(logr)
    grid_size = 100
    logm_grid = np.linspace(-3.522, 5.477, int(grid_size))

    hyper = pick_random_hyper(all_hyper, sample_size=sample_size)

    def func():
        return np.array([ ProbRGivenM(logr[i], logm_grid, hyper[i,:])
                         for i in range(sample_size)])
    
    benchmark(func)

@pytest.mark.probrbench
def test_probR_new(benchmark):
    from scipy.stats import norm, truncnorm
    from forecaster.func import generate_mass, pick_random_hyper, \
        piece_linear, piece_linear_II, ProbRGivenM_II

    all_hyper = load_file()
    mean = 0.01
    std = 0.001
    sample_size = NSAMPLES
    radius = truncnorm.rvs( (0.-mean)/std, np.inf, loc=mean, scale=std, size=sample_size)
    logr = np.log10(radius)
    logm = np.ones_like(logr)
    grid_size = 100
    logm_grid = np.linspace(-3.522, 5.477, int(grid_size))

    hyper = pick_random_hyper(all_hyper, sample_size=sample_size)
    
    benchmark(ProbRGivenM_II, logr, logm_grid, hyper)