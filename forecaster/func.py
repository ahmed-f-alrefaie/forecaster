import numpy as np
from scipy.stats import norm, truncnorm
from numpy.random import default_rng


### fix the number of different populations
n_pop = 4

def pick_random_hyper(all_hyper, sample_size=None):
	rng = default_rng()
	size = sample_size or all_hyper.shape[0]
	return rng.choice(all_hyper, size=sample_size, replace=False)





def indicate(M, trans, i):
	'''
	indicate which M belongs to population i given transition parameter
	'''
	ts = np.insert(np.insert(trans, n_pop-1, np.inf), 0, -np.inf)
	return (M>=ts[i]) & (M<ts[i+1])

def indicate_II(M, trans, i):

	return (M>=trans[...,i]) & (M<trans[...,i+1])



def split_hyper_linear(hyper):
	'''
	split hyper and derive c
	'''
	c0, slope,sigma, trans = \
	hyper[0], hyper[1:1+n_pop], hyper[1+n_pop:1+2*n_pop], hyper[1+2*n_pop:]

	c = np.zeros_like(slope)
	c[0] = c0
	for i in range(1,n_pop):
		c[i] = c[i-1] + trans[i-1]*(slope[i-1]-slope[i])

	return c, slope, sigma, trans


def split_hyper_linear_II(hyper):
	'''
	split hyper and derive c
	'''
	c0, slope,sigma, trans = \
	hyper[...,0], hyper[...,1:1+n_pop], hyper[...,1+n_pop:1+2*n_pop], hyper[...,1+2*n_pop:]

	c = np.zeros_like(slope)
	c[...,0] = c0
	for i in range(1,n_pop):
		c[...,i] = c[...,i-1] + trans[...,i-1]*(slope[...,i-1]-slope[...,i])
	trans = np.insert(np.insert(trans,n_pop-1,np.inf,axis=1), 0, -np.inf, axis=1)
	return c, slope, sigma, trans


def piece_linear_II(hyper, M, prob_R):
	c, slope, sigma, trans = split_hyper_linear_II(hyper)

	M = M

	R = np.zeros_like(M)

	for i in range(n_pop):
		ind = indicate_II(M, trans, i)
		mu = c[...,i]
		mu[ind] +=  M[ind]*slope[ind,i]
		R[ind] = norm.ppf(prob_R[ind],mu[ind],sigma[ind,i])

	return R

def generate_mass(mean, std, sample_size):
	mlower = 3e-4
	mupper = 3e5
	return truncnorm.rvs( (mlower-mean)/std, (mupper-mean)/std, loc=mean, scale=std, size=sample_size)	


def piece_linear(hyper, M, prob_R):
	'''
	model: straight line
	'''

	M = np.array(M)
	c, slope, sigma, trans = split_hyper_linear(hyper)
	R = np.zeros_like(M)

	for i in range(4):
		ind = indicate(M, trans, i)

		mu = c[i] + M[ind]*slope[i]
		R[ind] = norm.ppf(prob_R[ind], mu, sigma[i])

	return R


def ProbRGivenM(radii, M, hyper):
	'''
	p(radii|M)
	'''
	c, slope, sigma, trans = split_hyper_linear(hyper)
	prob = np.zeros_like(M)
	#print('SHAPE', prob.shape, M.shape, slope.shape)
	for i in range(4):
		ind = indicate(M, trans, i)
		#print('MSHAPE',M[ind].shape)
		mu = c[i] + M[ind]*slope[i]
		#print('EXPECTED',mu)
		sig = sigma[i]
		prob[ind] = norm.pdf(radii, mu, sig)

	prob = prob/np.sum(prob)

	return prob

def ProbRGivenM_II(radii, M, hyper):
	c, slope, sigma, trans = split_hyper_linear_II(hyper)
	# 10, 100
	prob = np.zeros(shape=(radii.shape[0], M.shape[0]))
	mu = np.zeros_like(prob)
	for i in range(n_pop):
		mu[...] = 0.0
		ind = indicate_II(M[None,...], trans[:,None,:], i)
		radii_id,mass_id = np.where(ind)
		#
		mu[radii_id, mass_id] = c[radii_id,i] + slope[radii_id,i]*M[mass_id]#M[None,...]*slope[:,None,i][ind]
		#print(mu[0])
		prob[ind] = norm.pdf(radii[radii_id],mu[radii_id, mass_id],sigma[radii_id,i])
	#print('C',c[:,None,i])
	return (prob/np.sum(prob, axis=1)[:,None])

def random_choice_2d(arr, probs):
	idx = (probs.cumsum(1) > np.random.rand(probs.shape[0])[:,None]).argmax(1)
	return arr[idx]



def classification( logm, trans ):
	'''
	classify as four worlds
	'''
	count = np.zeros(4)
	sample_size = len(logm)
	ts = np.insert(np.insert(trans, n_pop-1, np.inf), 0, -np.inf)
	for iclass in range(4):
		
		ind = indicate_II( logm, ts, iclass)
		count[iclass] = count[iclass] + ind.sum()
	
	prob = count / np.sum(count) * 100.
	print ('Terran %(T).1f %%, Neptunian %(N).1f %%, Jovian %(J).1f %%, Star %(S).1f %%' \
			% {'T': prob[0], 'N': prob[1], 'J': prob[2], 'S': prob[3]})
	return None