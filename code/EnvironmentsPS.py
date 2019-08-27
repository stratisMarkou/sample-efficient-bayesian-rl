import numpy as np

import torch

from utils import normal_gamma, solve_tabular_continuing_PI


# ============================================================================
# Environment class
# ============================================================================

class Environment:
    
    def __init__(self):
        pass
    
    
    def sample_dynamics(self, num_frames):
        '''
            Sample feasible space of dynamics (random state and action)
                num_frames : integer number of frames
        '''
        
        states, actions, rewards, states_ = [], [], [], []
        
        for i in range(num_frames):
            
            # Sample starting state and action
            internal_state = np.random.uniform(low=-self.internal_state_scales,
                                               high=self.internal_state_scales)
            action_normalised = np.random.uniform(low=-np.ones(shape=(self.action_dim,)),
                                                  high=np.ones(shape=(self.action_dim,)))
            action = self.action_from_agent_output(action_normalised)
            
            # Obtain next internal state
            next_internal_state, reward, _ = self.step(internal_state, action)
            
            states.append(self.observations_from_internal_states(internal_state))
            actions.append(action_normalised)
            rewards.append(reward)
            states_.append(self.observations_from_internal_states(next_internal_state))
        
        return [np.array(array) for array in (states, actions, rewards, states_)]
    
    
    def run_episode(self, s0, num_frames, agent, policy_params):
        '''
            Run one episode using
                s0         : numpy array initial (internal) state
                num_frames : integer number of frames
                agent      : Agent object
                agent_mode : string, e.g. 'random', 'greedy', 'e-greedy', 'ucb'
        '''
        
        internal_state = s0[:]
        states, actions, rewards, states_ = [], [], [], []
        
        for i in range(num_frames):
            
            # Get observed state and select action using the agent
            observed_state = self.observations_from_internal_states(internal_state)
            agent_action = agent.take_action(observed_state[None, :], policy_params=policy_params)[0]
            action = self.action_from_agent_output(agent_action=agent_action)
            
            # Evolve environment dynamics
            next_internal_state, reward, done = self.step(internal_state, action)
            next_observed_state = self.observations_from_internal_states(next_internal_state)
            
            # Store states, actions, rewards 
            states.append(observed_state)
            actions.append(agent_action)
            rewards.append(reward)
            states_.append(next_observed_state)
            
            # Update internal state
            internal_state = next_internal_state

            if done: break
        
        return [np.array(array) for array in (states, actions, rewards, states_)]
        
        
    
    def step(self, internal_state, action):
        '''
            Return next state and reward from current internal state and action
            and also the full trajectory between decision points, if available
        '''
        
        raise NotImplementedError
    
    
    def observations_from_internal_states(self, internal_state):
        '''
            Convert the internal state of the environment to the observations passed to the agent
                internal_state : numpy array in the range of the internal state
        '''
        
        raise NotImplementedError
        
        
    def action_from_agent_output(self, agent_action):
        '''
            Convert the action chosen by the agent [-1, 1] to environment dimensions
                agent_action : numpy array in [-1, 1] range
        '''
        
        return NotImplementedError
    
    
    def get_name(self):
        '''
            Get environment name for saving
        '''
        
        return NotImplementedError
    

# ============================================================================
# Tabular environment definition
# ============================================================================


class TabularEnvironment(Environment):

    def __init__(self, T):
        super(Environment, self).__init__()
        
        # Get custom-implemented dynamics and reward distributions
        self.P, self.R = self.get_dynamics_and_rewards_distributions()

        # Set current time to 0 and maximum time
        self.t = 0
        self.T = T
        self.done = False

    def get_dynamics_and_rewards_distributions(self):
        '''
            Returns the dynamics and rewards probability distributions.
            Both P(s, a) and R(s, a, s_) must be callable, and return the
            sampled next state s_ and reward r, respectively
        '''

        raise NotImplementedError


    def step(self, a):

        if self.done: raise Exception('Cannot step after end of episode!')

        # Sample next state
        s_ = self.P(self.s, a)

        # Sample reward
        r = self.R(self.s, a, s_)

        # Check if episode limit has been reached
        self.done = True if self.t >= self.T else False

        # Add one to time count
        self.t += 1
        self.s = s_

        return s_, r, self.t, self.done


    def reset(self):
        self.t = 0
        self.s = 0
        self.done = False


    def sa_list(self):
        '''
            Returns a list of all (s, a) permissible state-action combinations
        '''

        sa_list = []

        for sa, _ in self.P_probs.items():
            sa_list.append(sa)

        return sa_list


    def observations_from_internal_state(self, internal_state):
        return internal_state


    def action_from_agent_output(self, agent_action):
        return agent_action


    def get_optimal_VQpi(self):
        raise NotImplementedError


# ============================================================================
# RiverSwim environment
# ============================================================================


class RiverSwim(TabularEnvironment):

    def __init__(self, L, T):

        # Length of river
        self.L = L

        super(RiverSwim, self).__init__(T)


    def get_dynamics_and_rewards_distributions(self):
        '''
            Implementation of the corresponding method from TabularEnvironment
            Action 0 is left, 1 is right. There are N states and they all look
            the same except the ones on the boundaries.
        '''

        # Dict for transitions, P_probs[(s, a)] = [(s1, ...), (p1, ...)]
        P_probs = {}
        
        for s in range(1, self.L - 1):

            # Probabilities associated with swimming left
            P_probs[(s, 0)] = [(s - 1,), (1.00,)]

            # Probabilities associated with swimming right
            P_probs[(s, 1)] = [(s - 1, s, s + 1), (0.05, 0.35, 0.60)]
 
        # Probabilities for start state
        P_probs[(0, 0)] = [(0,), (1.00,)]
        P_probs[(0, 1)] = [(0, 1), (0.40, 0.60)]

        # Probabilities for end state 
        P_probs[(self.L - 1, 0)] = [(self.L - 2,), (1.00,)]
        P_probs[(self.L - 1, 1)] = [(self.L - 1, self.L - 2), (0.40, 0.60)] 

        def P(s, a):
            
            # States accessible from (s, a)
            s_, p = P_probs[(s, a)]

            # Sample s_ according to the transition probabilities
            s_ = np.random.choice(s_, p=p)

            return s_

        def R(s, a, s_):

            if s_ == 0:
                return 5e-3
            elif s_ == self.L - 1:
                return 1
            else:
                return 0

        self.P_probs = P_probs

        return P, R


# ============================================================================
# DeepSea environment
# ============================================================================


class DeepSea(TabularEnvironment):

    def __init__(self, params):
        

        assert params['L'] > 2, 'Sea must be larger than L = 2!'

        # Size of sea
        self.L = params['L']

        # Means and scales for rewards of each transition
        rew_params = params['rew_params']
        self.mu_l, self.sig_l = rew_params[0]
        self.mu_r, self.sig_r = rew_params[1]
        self.mu_t, self.sig_t = rew_params[2]
        
        T = L if params['episodic'] else float('inf')

        super(DeepSea, self).__init__(T=T)

    
    def get_dynamics_and_rewards_distributions(self):
        '''
            Implementation of the corresponding method from TabularEnvironment
        '''

        L = self.L
        
        # Dict for transitions, P_probs[(s, a)] = [(s1, ...), (p1, ...)]
        P_probs = {}
   
        for l in range(self.L - 1):

            # Swimming left
            P_probs[(l, 0)] = [(max(l - 1, 0),), (1.00,)]

            # Swimming right
            P_probs[(l, 1)] = [(max(l - 1, 0), l + 1), (1 / L, 1 - 1 / L)]

        P_probs[(self.L - 1, 0)] = [(L - 2,), (1.00,)]
        P_probs[(self.L - 1, 1)] = [(L - 2, 0), (1 / L, 1 - 1 / L)]
        
        self.P_probs = P_probs

        def P(s, a):
            
            # States accessible from (s, a)
            s_, p = P_probs[(s, a)]

            # Sample s_ according to the transition probabilities
            s_ = np.random.choice(s_, p=p)

            return s_


        def R(s, a, s_):

            rnd = np.random.rand()

            if s == self.L - 1 and a == 1 and s_ == 0:
                return self.mu_t + self.sig_t * rnd
            elif a == 1:
                return self.mu_r + self.sig_r * rnd
            else:
                return self.mu_l + self.sig_l * rnd

        return P, R
    
    
    def get_name(self):

        mode = 'E' if self.T == float('inf') else 'C'
        placeholder = 'DeepSea-{}-L_{}-mul_{}-mur_{}-mut_{}-sigl_{}-sigr_{}-sigt_{}'
        name = placeholder.format(mode, self.L, self.mu_l, self.mu_r, self.mu_t, self.sig_l, self.sig_r, self.sig_t)

        return name
    
    
    def get_P_and_R(self):
        
        P = np.zeros((self.L, 2, self.L))
        R = np.zeros((self.L, 2, self.L))
        
        for s in range(self.L):
            for a in range(2):
                
                next_states = self.P_probs[(s, a)][0]
                probs = self.P_probs[(s, a)][1]
                
                for s_ in range(self.L):
                    
                    P[s, a, s_] = probs[next_states.index(s_)] if s_ in next_states else 0.
                    
                    # Large negative penalty for non-allowed transitions
                    R[s, a, s_] = -1e6
                    
                    # Swim left and hit wall
                    if s == 0 and s_ == 0 and a == 0:
                        R[s, a, s_] = self.mu_l
                        
                    # Swim right but move left and hit wall
                    elif s == 0 and s_ == 0 and a == 1:
                        R[s, a, s_] = self.mu_r
                    
                    # Swim right and get treasure
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
                
            
    def get_optimal_policy(self, gamma, num_iter):
        
        P, R = self.get_P_and_R()
        
        pi, Q = solve_tabular_continuing_PI(P, R, gamma, num_iter)
        
        return pi, Q
            


    
# ============================================================================
# WideNarrow environment
# ============================================================================


class WideNarrow(TabularEnvironment):

    def __init__(self, params):
        
        N = params['N']
        W = params['W']
        episodic = params['episodic']
        rew_params = params['rew_params']
        

        # Number of WideNarrow compartments and width of wide parts
        self.N, self.W = N, W

        # Means and scales for rewards of each transition
        self.mu_l, self.sig_l = rew_params[0]
        self.mu_h, self.sig_h = rew_params[1]
        self.mu_n, self.sig_n = rew_params[2]
        
        T = 2 * self.N if episodic else float('inf')

        super(WideNarrow, self).__init__(T)

    
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
            
            # States accessible from (s, a)
            s_, p = P_probs[(s, a)]

            # Sample s_ according to the transition probabilities
            s_ = np.random.choice(s_, p=p)

            return s_


        def R(s, a, s_):

            even_s = s % 2 == 0
            odd_s_ = s_ % 2 == 1

            if s == 2 * self.N and s_ == 0:
                return 0.
            elif even_s and odd_s_ and (a == 0):
                return self.mu_h + self.sig_h * np.random.normal()
            elif even_s and odd_s_:
                return self.mu_l + self.sig_l * np.random.normal()
            else:
                return self.mu_n + self.sig_n * np.random.normal()

        return P, R
    
    
    def get_name(self):

        mode = 'E' if self.T == float('inf') else 'C'
        placeholder = 'WideNarrow-{}_N-{}_W-{}_mul-{}_muh-{}_mun-{}_sigl-{}_sigh-{}_sign-{}'
        name = placeholder.format(mode, self.N, self.W, self.mu_l, self.mu_h, self.mu_n,
                                  self.sig_l, self.sig_h, self.sig_n)

        return name
    
    
    def get_P_and_R(self):
        
        P = np.zeros((2 * self.N + 1, self.W, 2 * self.N + 1))
        R = np.zeros((2 * self.N + 1, self.W, 2 * self.N + 1))
        
        for s in range(2 * self.N + 1):
            for a in range(self.W):
                
                if not((s, a) in self.P_probs):
                    P[s, a, :] = 1. / (2 * self.N + 1)
                
                for s_ in range(2 * self.N + 1):
                    
                    # Large negative penalty for non-allowed transitions
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

        num_s = params['num_s']
        num_a = params['num_a']
        episodic = params['episodic']
        kappa = params['kappa']
        mu = params['mu']
        lamda = params['lamda']
        alpha = params['alpha']
        beta = params['beta']
        seed = params['seed']
        
        # Size of MDP
        self.num_s = num_s
        self.num_a = num_a

        # Means and scales for rewards of each transition
        self.kappa = kappa * np.ones((num_s,))
        self.mu = mu
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        
        T = num_s if episodic else float('inf')

        super(PriorMDP, self).__init__(T)

    
    def get_dynamics_and_rewards_distributions(self):
        '''
            Implementation of the corresponding method from TabularEnvironment
        '''

        # Shorter names for constants
        states = np.arange(self.num_s)
        actions = np.arange(self.num_a)

        mu = self.mu
        lamda = self.lamda
        alpha = self.alpha
        beta = self.beta

        # Dict for transitions, P_probs[(s, a)] = [(s1, ...), (p1, ...)]
        P_probs = {}
        R_mu_prec = {}

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

        mode = 'E' if self.T == float('inf') else 'C'

        placeholder = 'PriorMDP-{}_s-{}_a-{}_kappa-{}_mu-{}_lamda-{}_alpha-{}_beta-{}_seed-{}'

        name = placeholder.format(mode, self.num_s,  self.num_a, self.kappa[0], self.mu,
                                  self.lamda, self.alpha, self.beta, self.seed)
        return name
    
    
    def get_optimal_policy(self, gamma, num_iter):
        
        P, R = self.get_P_and_R()
        
        pi, Q = solve_tabular_continuing_PI(P, R, gamma, num_iter)
        
        return pi, Q
    
