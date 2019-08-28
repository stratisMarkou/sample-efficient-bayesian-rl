from numpy.random import normal, gamma

from pynverse import inversefunc

from scipy.special import digamma

# ============================================================================
# util definitions
# ============================================================================


def bql_f_inv(x):
    """
	Returns the inverse of f at x where:
	    f(x) = log(x) - digamma(x)
    """    

    # Function to take the inverse of
    def bql_f(x_):
        return np.log(x_) - digamma(x_)

    result = inversefunc(bql_f,
			 y_values=x,
			 domain=[1e-12, 1e12],
			 open_domain=True,
			 image=[1e-16, 1e16])

    return float(result)


def normal_gamma(mu0, lamda, alpha, beta):
    """
	Returns samples from Normal-Gamma with the specified parameters.
	
	Number of samples returned is the length of mu0, lambda, alpha, beta.
    """    

    # Check if parameters are scalars or vetors
    if type(mu0) == float:
        size = (1,)
    else:
        size = mu0.shape
        
    # Draw samples from gamma (numpy "scale" is reciprocal of beta)
    taus = gamma(shape=alpha, scale=beta**-1, size=size)

    # Draw samples from normal condtioned on the sampled precision
    mus = normal(loc=mu0, scale=(lamda * taus)**-0.5, size=size)
    
    return mus, taus


def solve_tabular_continuing_VI(P, R, gamma, tolerance, max_iter):
    '''
        Solves the Bellman equation for a continuing tabular problem.

	Returns greedy policy pi and corresponding Q-values.
    '''
    
    num_s, num_a = P.shape[:2]
    
    # Initialise Q, pi
    Q = np.zeros((num_s, num_a))
    Q_next = Q[:, :]
    pi = None

    for i in range(num_iter):

	# Compute Sum_{s'} P(s, a, s') max_{a'}[Q(s', a')]	
	Q_max = np.max(Q, axis=-1)
        P_Q_max = np.einsum('ijk, k -> ij', P, Q_max)

	# Compute Sum_{s'} P(s, a, s') R(s, a, s')
        P_R = np.einsum('ijk, ijk -> ij', P, R)

	# Q values are the sum of immediate rewards and discounted next Q.
        Q_next = P_R + gamma * P_Q_max

	# If change in Q below tolerance, stop
	if np.abs(Q_max - np.max(Q_next, axis=-1)) < tolerance:
	    break

	Q = Q_next

    # Get greedy policy, break ties at random to avoid biasing choice
    pi = np.array([np.random.choice(np.argwhere(Qs == np.amax(Qs))[:, 0]) \
	           for Qs in Q])

    return pi, Q
