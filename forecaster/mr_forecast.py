import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm 
import h5py 
import pkg_resources
from functools import lru_cache
from forecaster.func import pick_random_hyper, piece_linear, ProbRGivenM, classification, piece_linear_II, ProbRGivenM_II, random_choice_2d
## constant
mearth2mjup = 317.828
mearth2msun = 333060.4
rearth2rjup = 11.21
rearth2rsun = 109.2

## boundary
mlower = 3e-4
mupper = 3e5

## number of category
n_pop = 4

@lru_cache(maxsize=2)
def load_file():
	hyperfile_path = pkg_resources.resource_filename(
        'forecaster', 'data/fitting_parameters.h5')
	with h5py.File(hyperfile_path, 'r') as f:
		all_hyper = f['hyper_posterior'][...]
	return all_hyper

## function


##############################################

def Mpost2R(mass, unit='Earth', classify='No'):
	"""
	Forecast the Radius distribution given the mass distribution.

	Parameters
	---------------
	mass: one dimensional array
		The mass distribution.
	unit: string (optional)
		Unit of the mass. 
		Options are 'Earth' and 'Jupiter'. Default is 'Earth'.
	classify: string (optional)
		If you want the object to be classifed. 
		Options are 'Yes' and 'No'. Default is 'No'.
		Result will be printed, not returned.

	Returns
	---------------
	radius: one dimensional array
		Predicted radius distribution in the input unit.
	"""

	all_hyper = load_file()
	# mass input
	mass = np.array(mass)
	assert len(mass.shape) == 1, "Input mass must be 1-D."

	# unit input
	if unit == 'Earth':
		pass
	elif unit == 'Jupiter':
		mass = mass * mearth2mjup
	else:
		print ("Input unit must be 'Earth' or 'Jupiter'. Using 'Earth' as default.")

	# mass range
	if np.min(mass) < 3e-4 or np.max(mass) > 3e5:
		print ('Mass range out of model expectation. Returning None.')
		return None

	## convert to radius
	sample_size = len(mass)
	logm = np.log10(mass)
	prob = np.random.random(sample_size)
	logr = np.ones_like(logm)

	hyper = pick_random_hyper(all_hyper, sample_size=sample_size)
	if classify == 'Yes':
		classification(logm, hyper[:,-3:])

	logr = piece_linear_II(hyper, logm, prob)
	# for i in range(sample_size):
	# 	logr[i] = piece_linear(hyper[i], logm[i], prob[i])

	radius_sample = 10.** logr

	return radius_sample / rearth2rjup if unit == 'Jupiter' else radius_sample



def Mstat2R(mean, std, unit='Earth', sample_size=1000, classify = 'No'):	
	"""
	Forecast the mean and standard deviation of radius given the mena and standard deviation of the mass.
	Assuming normal distribution with the mean and standard deviation truncated at the mass range limit of the model.

	Parameters
	---------------
	mean: float
		Mean (average) of mass.
	std: float
		Standard deviation of mass.
	unit: string (optional)
		Unit of the mass. Options are 'Earth' and 'Jupiter'.
	sample_size: int (optional)
		Number of mass samples to draw with the mean and std provided.
	Returns
	---------------
	mean: float
		Predicted mean of radius in the input unit.
	std: float
		Predicted standard deviation of radius.
	"""

	# unit
	if unit == 'Earth':
		pass
	elif unit == 'Jupiter':
		mean = mean * mearth2mjup
		std = std * mearth2mjup
	else:
		print("Input unit must be 'Earth' or 'Jupiter'. Using 'Earth' as default.")

	# draw samples
	mass = truncnorm.rvs( (mlower-mean)/std, (mupper-mean)/std, loc=mean, scale=std, size=sample_size)	
	if classify == 'Yes':	
		radius = Mpost2R(mass, unit='Earth', classify='Yes')
	else:
		radius = Mpost2R(mass, unit='Earth')

	if unit == 'Jupiter':
		radius = radius / rearth2rjup

	r_med = np.median(radius)
	onesigma = 34.1
	r_up = np.percentile(radius, 50.+onesigma, interpolation='nearest')
	r_down = np.percentile(radius, 50.-onesigma, interpolation='nearest')

	return r_med, r_up - r_med, r_med - r_down



def Rpost2M(radius, unit='Earth', grid_size = 1e3, classify = 'No'):
	"""
	Forecast the mass distribution given the radius distribution.

	Parameters
	---------------
	radius: one dimensional array
		The radius distribution.
	unit: string (optional)
		Unit of the mass. Options are 'Earth' and 'Jupiter'.
	grid_size: int (optional)
		Number of grid in the mass axis when sampling mass from radius.
		The more the better results, but slower process.
	classify: string (optional)
		If you want the object to be classifed. 
		Options are 'Yes' and 'No'. Default is 'No'.
		Result will be printed, not returned.

	Returns
	---------------
	mass: one dimensional array
		Predicted mass distribution in the input unit.
	"""
	

	all_hyper = load_file()
	# unit
	if unit == 'Earth':
		pass
	elif unit == 'Jupiter':
		radius = radius * rearth2rjup
	else:
		print ("Input unit must be 'Earth' or 'Jupiter'. Using 'Earth' as default.")


	# radius range
	if np.min(radius) < 1e-1 or np.max(radius) > 1e2:
		print ('Radius range out of model expectation. Returning None.')
		return None



	# sample_grid
	if grid_size < 10:
		print ('The sample grid is too sparse. Using 10 sample grid instead.')
		grid_size = 10

	## convert to mass
	sample_size = len(radius)
	logr = np.log10(radius)
	logm = np.ones_like(logr)

	hyper_ind = np.random.randint(low = 0, high = np.shape(all_hyper)[0], size = sample_size)
	hyper = all_hyper[hyper_ind,:]

	logm_grid = np.linspace(-3.522, 5.477, int(grid_size))

	prob = ProbRGivenM_II(logr, logm_grid, hyper)
	logm = random_choice_2d(logm_grid, prob)

	mass_sample = 10.** logm

	if classify == 'Yes':
		classification(logm, hyper[:,-3:])

	return mass_sample / mearth2mjup if unit == 'Jupiter' else mass_sample



def Rstat2M(mean, std, unit='Earth', sample_size=1e3, grid_size=1e3, classify = 'No'):	
	"""
	Forecast the mean and standard deviation of mass given the mean and standard deviation of the radius.

	Parameters
	---------------
	mean: float
		Mean (average) of radius.
	std: float
		Standard deviation of radius.
	unit: string (optional)
		Unit of the radius. Options are 'Earth' and 'Jupiter'.
	sample_size: int (optional)
		Number of radius samples to draw with the mean and std provided.
	grid_size: int (optional)
		Number of grid in the mass axis when sampling mass from radius.
		The more the better results, but slower process.
	Returns
	---------------
	mean: float
		Predicted mean of mass in the input unit.
	std: float
		Predicted standard deviation of mass.
	"""
	# unit
	if unit == 'Earth':
		pass
	elif unit == 'Jupiter':
		mean = mean * rearth2rjup
		std = std * rearth2rjup
	else:
		print("Input unit must be 'Earth' or 'Jupiter'. Using 'Earth' as default.")

	# draw samples
	radius = truncnorm.rvs( (0.-mean)/std, np.inf, loc=mean, scale=std, size=sample_size)	
	if classify == 'Yes':
		mass = Rpost2M(radius, 'Earth', grid_size, classify='Yes')
	else:
		mass = Rpost2M(radius, 'Earth', grid_size)

	if mass is None:
		return None

	if unit=='Jupiter':
		mass = mass / mearth2mjup

	m_med = np.median(mass)
	onesigma = 34.1
	m_up = np.percentile(mass, 50.+onesigma, interpolation='nearest')
	m_down = np.percentile(mass, 50.-onesigma, interpolation='nearest')

	return m_med, m_up - m_med, m_med - m_down


	