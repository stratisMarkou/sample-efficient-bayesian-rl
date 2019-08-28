import numpy as np

import torch

from utils import normal_gamma, solve_tabular_continuing_VI


# ============================================================================
# Tabular environment definition
# ============================================================================


class TabularEnvironment:

    def __init__(self, T):
        
        # Get custom-implemented dynamics and reward distributions
        self.P, self.R = self.get_dynamics_and_rewards_distributions()
    

    def run_episode(self, s0, num_steps, agent, policy_params):
        '''
            Run one episode using
                s0         : numpy array initial state
                num_steps  : integer number of time steps
                agent      : Agent object
        '''
        
        s = s0[:]
        states, actions, rewards, states_ = [], [], [], []
	data = [states, actions, rewards, states_]
        
        for i in range(num_steps):
            
            # Get observed state and select action using the agent
            a = agent.take_action(s, policy_params=policy_params)[0]
            
            # Evolve environment dynamics
            s_, r = self.step(s, a)
            
            # Store states, actions, rewards 
	    for l, entry in zip(data, [s, a, r, s_]): l.append(entry)
            
            # Update internal state
            s = s_

        return [np.array(l) for l in data]
        
        
    def step(self, a):

        # Sample next state
        s_ = self.P(self.s, a)

        # Sample reward
        r = self.R(self.s, a, s_)

	# Set new state
        self.s = s_

        return s_, r
 

    def reset(self):
        self.s = 0


    def sa_list(self):
        '''
            Returns list of (s, a), permissible state-action combinations. 
        '''

        sa_list = []

        for sa, _ in self.P_probs.items():
            sa_list.append(sa)

        return sa_list
     

    def get_optimal_policy(self, gamma, tolerance, max_iter):
	"""
	    Get optimal policy by value iteration.
	"""        

        P, R = self.get_P_and_R()
        
        pi, Q = solve_tabular_continuing_VI(P, R, gamma, tolerance, max_iter)
        
        return pi, Q
        

    def get_dynamics_and_rewards_distributions(self):
        '''
	    Must be implemented by child class.

            Returns callables P and R, the dynamics and reward distributions.
        '''

        raise NotImplementedError


    def get_name(self):
        '''
            Get environment name for saving.
        '''
        
        return NotImplementedError


# ============================================================================
# DeepSea environment
# ============================================================================


class DeepSea(TabularEnvironment):

    def __init__(self, params):
        
	valid_N = params['N'] > 2 and params['N'] < 40
        assert valid_N, 'DeepSea requires 2 < N < 40!'

        # DeepSea parameters
        self.N = params['N']
        rew_params = params['rew_params']
        self.mu_l, self.sig_l = rew_params[0]
        self.mu_r, self.sig_r = rew_params[1]
        self.mu_t, self.sig_t = rew_params[2]
        
        super(DeepSea, self).__init__()
    

    def get_dynamics_and_rewards_distributions(self):
        '''
            Implementation of the corresponding method from TabularEnvironment
        '''

        N = self.N
        
        # Dictionary for transitions, P_probs[(s, a)] = [(s1, ...), (p1, ...)]
        P_probs = {}
   
        for n in range(N - 1):

            # Swimming left
            P_probs[(n, 0)] = [(max(n - 1, 0),), (1.00,)]

            # Swimming right
            P_probs[(n, 1)] = [(max(n - 1, 0), n + 1), (1 / N, 1 - 1 / N)]

	# Swimming left from last state
        P_probs[(N - 1, 0)] = [(N - 2,), (1.00,)]

	# Swimming right from last state
        P_probs[(N - 1, 1)] = [(N - 2, 0), (1 / N, 1 - 1 / N)]
        
        self.P_probs = P_probs

        def P(s, a):
            
            # Next states and transition probabilities
            s_, p = P_probs[(s, a)]

            # Sample s_ and return
            return np.random.choice(s_, p=p)

        def R(s, a, s_):

            rnd = np.random.rand()

	    # Successful swim-right from last state 
            if s == N - 1 and a == 1 and s_ == 0:
                return self.mu_t + self.sig_t * rnd
	    # All other swim-rights
            elif a == 1:
                return self.mu_r + self.sig_r * rnd
	    # All swim-lefts
            else:
                return self.mu_l + self.sig_l * rnd

        return P, R
    

    def get_name(self):
	""" Returns environment name for saving. """
        return 'DeepSea-N_{}'.format(N)


    def get_mean_P_and_R(self):
	""" Returns true P and expected R for solving optimal policy. """

        P = np.zeros((self.L, 2, self.L))
        R = np.zeros((self.L, 2, self.L))
        
        for s in range(self.L):
            for a in range(2):
                for s_ in range(self.N):
 
		    if s_ in self.P_probs[(s, a)][0]:
			P[s, a, s_] = probs[next_states.index(s_)]
		    else:
			P[s, a, s_] = 0.
                    
                    # Large negative penalty for non-allowed transitions (hack)
                    R[s, a, s_] = -1e6
                    
                    # Swim left and hit wall
                    if s == 0 and s_ == 0 and a == 0:
                        R[s, a, s_] = self.mu_l
                        
                    # Swim right but move left and hit wall
                    elif s == 0 and s_ == 0 and a == 1:
                        R[s, a, s_] = self.mu_r
                    
                    # Swim right and get high reward
                    elif s == self.L - 1 and a == 1 and s_ == 0:
                        R[s, a, s_] = self.mu_t
                    
                    # Swim right but move left
                    elif s_ == s - 1 and a == 1:
                        R[s, a, s_] = self.mu_r
                    
                    # Swim left and move left
                    elif s_ == s - 1 and a == 0:
                        R[s, a, s_] = self.mu_l
                    
                    # Swim right and move right
                    elif s_ == s + 1 and a == 1:
                        R[s, a, s_] = self.mu_r
                        
        return P, R
                

           
# ============================================================================
# WideNarrow environment
# ============================================================================


class WideNarrow(TabularEnvironment):

    def __init__(self, params):
        
	# WideNarrow parameters
        self.N, self.W = params['N'], params['W']
        self.mu_l, self.sig_l = params['rew_params'][0]
        self.mu_h, self.sig_h = params['rew_params'][1]
        self.mu_n, self.sig_n = params['rew_params'][2]
        
        super(WideNarrow, self).__init__()

    
    def get_dynamics_and_rewards_distributions(self):
        '''
            Implementation of the corresponding method from TabularEnvironment
        '''

        # Dict for transitions, P_probs[(s, a)] = [(s1, ...), (p1, ...)]
        P_probs = {}
        
        for n in range(self.N):
            for a in range(self.W):

                # Wide part transitions
                P_probs[(2 * n, a)] = [(2 * n + 1,), (1.00,)]
            
            P_probs[(2 * n + 1, 0)] = [(2 * n + 2,), (1.00,)]

        # Last state transitions to first state
        P_probs[(2 * self.N, 0)] = [(0,), (1.00,)]
         
        self.P_probs = P_probs
    
        def P(s, a):
            
            # Next states and transition probabilities 
            s_, p = P_probs[(s, a)]

            # Sample s_ according to the transition probabilities
            s_ = np.random.choice(s_, p=p)

            return s_


        def R(s, a, s_):

	    # Booleans for current and next state
            even_s, odd_s_ = s % 2 == 0, s_ % 2 == 1

	    # Zero reward for transition from last to first state
            if s == 2 * self.N and s_ == 0:
                return 0.
	    # High reward for correct action from odd state
            elif even_s and odd_s_ and (a == 0):
                return self.mu_h + self.sig_h * np.random.normal()
	    # Low reward for incorrect action from odd state
            elif even_s and odd_s_:
                return self.mu_l + self.sig_l * np.random.normal()
	    # Reward from even state
            else:
                return self.mu_n + self.sig_n * np.random.normal()

        return P, R
    
    
    def get_name(self):
        return 'WideNarrow-N-{}_W-{}'.format(self.N, self.W)
    
    
    def get_P_and_R(self):
        
        P = np.zeros((2 * self.N + 1, self.W, 2 * self.N + 1))
        R = np.zeros((2 * self.N + 1, self.W, 2 * self.N + 1))
        
        for s in range(2 * self.N + 1):
            for a in range(self.W):
                
		# Uniform prob hack for dissallowed states
                if not((s, a) in self.P_probs):
                    P[s, a, :] = 1. / (2 * self.N + 1)
                
                for s_ in range(2 * self.N + 1):
                    
                    # Large negative reward for non-allowed transitions
                    R[s, a, s_] = -1e6
                        
                    # Take action from last state
                    if s == 2 * self.N and s_ == 0 and a == 0:
                        P[s, a, s_] = 1.
                        R[s, a, s_] = 0.
                        
                    # Take good action from even state
                    elif s % 2 == 0 and s_ == s + 1 and a == 0:
                        P[s, a, s_] = 1.
                        R[s, a, s_] = self.mu_h
                        
                    # Take suboptimal action from even state
                    elif s % 2 == 0 and s_ == s + 1:
                        P[s, a, s_] = 1.
                        R[s, a, s_] = self.mu_l
                        
                    # Take action from odd state
                    elif s % 2 == 1 and s_ == s + 1 and a == 0:
                        P[s, a, s_] = 1.
                        R[s, a, s_] = self.mu_n
                        
        return P, R
                
            
    def get_optimal_policy(self, gamma, num_iter):
        
        P, R = self.get_P_and_R()
        
        pi, Q = solve_tabular_continuing_PI(P, R, gamma, num_iter)
        
        return pi, Q



# ============================================================================
# PriorMDP definition
# ============================================================================


class PriorMDP(TabularEnvironment):

    def __init__(self, params):

	# PriorMDP parameters
        self.num_s = params['num_s']
        self.num_a = params['num_a']
        self.kappa = params['kappa']
        self.mu = params['mu']
        self.lamda = params['lamda']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.seed = params['seed']

        super(PriorMDP, self).__init__()

    
    def get_dynamics_and_rewards_distributions(self):
        '''
            Implementation of the corresponding method from TabularEnvironment
        '''

	# Short names for constants
        states = np.arange(self.num_s)
        actions = np.arange(self.num_a)
        mu = self.mu
        lamda = self.lamda
        alpha = self.alpha
        beta = self.beta

        # Dict for transitions, P_probs[(s, a)] = [(s1, ...), (p1, ...)]
        P_probs = {}
        R_mu_prec = {}

	# Set random seed for sampling PriorMDP
        np.random.seed(self.seed)
        
        for s in states:
            for a in actions:
                P_probs[(s, a)] = [states, np.random.dirichlet(self.kappa)] 
            
                for s_ in states:
                    mu, prec = normal_gamma(mu, lamda, alpha, beta)
                    R_mu_prec[(s, a, s_)] = [mu, prec]

        self.P_probs = P_probs
        self.R_mu_prec = R_mu_prec

        def P(s, a):
            
            # States accessible from (s, a)
            s_, p = P_probs[(s, a)]

            # Sample s_ according to the transition probabilities
            s_ = np.random.choice(s_, p=p)

            return s_


        def R(s, a, s_):
            
            # Get mean and precision of rewards
            mu, prec = R_mu_prec[(s, a, s_)]

            return np.random.normal(loc=mu, scale=prec**-0.5)

        return P, R
    
    
    def get_P_and_R(self):
        
        P = np.zeros((self.num_s, self.num_a, self.num_s))
        R = np.zeros((self.num_s, self.num_a, self.num_s))
        
        for s in range(self.num_s):
            for a in range(self.num_a):
                P[s, a, :] = self.P_probs[(s, a)][1]
                
                for s_ in range(self.num_s):
                    R[s, a, s_] = self.R_mu_prec[(s, a, s_)][0]
                    
        return P, R
    
    
    def get_name(self):

        return 'PriorMDP-s_{}_a-{}-seed_{}'.format(self.num_s, self.num_a)

