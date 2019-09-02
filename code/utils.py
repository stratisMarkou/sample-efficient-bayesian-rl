import os
import pathlib
import pickle

import numpy as np
from numpy.random import normal, gamma

from pynverse import inversefunc

from scipy.special import digamma

from tqdm import tqdm_notebook as tqdm


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


def solve_tabular_continuing_PI(P, R, gamma, max_iter):
    '''
        Solves the Bellman equation for a continuing tabular problem.

        Returns greedy policy pi and corresponding Q-values.
    '''
    
    num_s, num_a = P.shape[:2]
    s_idx = np.arange(num_s)
    
    ones = np.eye(num_s)
    pi = np.zeros(num_s, dtype=np.int)
    Q = None
    
    P_R = np.einsum('ijk, ijk -> ij', P, R)
    
    for i in range(max_iter):
    
        # Solve for Q values
        V = np.linalg.solve(ones - gamma * P[s_idx, pi, :], P_R[s_idx, pi])
        Q = P_R + gamma * np.einsum('ijk, k -> ij', P, V)

        # Get greedy policy - break ties at random
        pi = np.array([np.random.choice(np.argwhere(Qs == np.amax(Qs))[0]) \ 
                       for Qs in Q])

    return pi, Q



# ============================================================================
# Experiment helpers
# ============================================================================


def run_experiment(environment,
                   agent,
                   seed,
                   num_time_steps,
                   max_buffer_length,
                   save_every):
    
    # Location to save agent
    save_loc = 'results/agent_logs/{}/'.format(environment.get_name())
    
    pathlib.Path(save_loc).mkdir(parents=True, exist_ok=True)
    
    # Set random seed and reset environment
    np.random.seed(seed)
    environment.reset()
    s, t = 0, 0
    
    agent_copies = []
    
    for i in tqdm(range(num_time_steps + 1)):
            
        # Take action
        a = agent.take_action(s, t)

        # Step environment
        s_, r, t = environment.step(a)

        # Update agent
        agent.observe([t, s, a, r, s_])
        agent.update_after_step(max_buffer_length, log=((i % save_every) == 0))

        # Update current state
        s = s_

    agent.save_copy(save_loc, agent.get_name() + '_seed-{}'.format(seed))
    
    
    
def run_oracle_experiment(environment,
                          seed,
                          gamma,
                          num_time_steps,
                          num_PI_iter):
    
    np.random.seed(seed)
    
    # Initial state
    environment.reset()
    s, t = 0, 0
    
    # Solve for optimal policy and corresponding Q
    P, R = environment.get_mean_P_and_R()
    pi, Q = solve_tabular_continuing_PI(P, R, gamma=gamma, max_iter=num_PI_iter)
    
    states, actions, rewards, states_ = [], [], [], []
    
    for i in range(num_time_steps + 1):

        # Take action
        a = pi[s]

        # Step environment
        s_, r, t = environment.step(a)
        
        # Update logging lists
        for l, entry in zip([states, actions, rewards, states_], [s, a, r, s_]): l.append(entry)

        # Update current state (for agent)
        s = s_
        
    return np.array(states), np.array(actions), np.array(rewards), np.array(states_)


def load_agent(environment, agent, seed):

    # Location to load from
    load_name = 'results/agent_logs/{}/{}_seed-{}'.format(environment.get_name(),
                                                          agent.get_name(),
                                                          seed)
    
    # Load the agent
    fhandle = open(load_name, 'rb')
    agent = pickle.load(fhandle)
    
    return agent