import numpy as np
import matplotlib.pyplot as plt

import pickle
import os

from scipy.special import digamma

from pynverse import inversefunc

from utils import phi,                          \
                  normal_gamma,                 \
                  solve_tabular_episodic_DP,    \
                  solve_tabular_continuing_PI


# ============================================================================
# General Tabular agent class
# ============================================================================


class TabularAgent:
    
    def __init__(self, gamma, T):

        # Discount factor
        self.gamma = gamma
        
        # Time horizon
        self.T = T
        
        
    def add_observations(self, s, a, r, s_):
        
        s, a, r, s_ = [np.array([thing]) for thing in [s, a, r, s_]]
        
        if hasattr(self, 'train_s'):
            self.train_s = np.concatenate([self.train_s, s], axis=0)
            self.train_a = np.concatenate([self.train_a, a], axis=0)
            self.train_s_ = np.concatenate([self.train_s_, s_], axis=0)
            self.train_r = np.concatenate([self.train_r, r], axis=0)
        
        else:
            self.train_s = s
            self.train_a = a
            self.train_s_ = s_
            self.train_r = r
            
            
    def take_action(self, s, t, policy_params):
        
        raise NotImplementedError
        
        
    def update_after_step(self, t):
        pass
    
    
    def update_after_episode(self):
        pass
    
    
    def observe(self, transition):
        pass
        
       
    def do_before_save(self):
        pass
        
    def save_copy(self, location, chkpt_name):
        
        self.do_before_save()
        
        fhandle = open(location + '/chkpt_{}'.format(chkpt_name), 'wb')
        pickle.dump(self, fhandle)
        fhandle.close()
        
        
# ============================================================================
# QLearningAgent class
# ============================================================================


class QLearningAgent(TabularAgent):
    
    def __init__(self, params):

        gamma = params['gamma']
        lr = params['lr']
        sa_list = params['sa_list']
        Q0 = params['Q0']
        T = params['T']
        dither_mode = params['dither_mode']
        dither_param = params['dither_param']
        anneal_timescale = params['anneal_timescale']

        super(QLearningAgent, self).__init__(gamma, T)
        
        # Set dithering mode (epsilon-greedy) or (boltzmann)
        self.dither_mode = dither_mode
        self.dither_param = dither_param
        self.anneal_timescale = anneal_timescale

        # Set learning rate schedule (this is a function)
        self.lr = lr
        
        # Q function initialisation
        self.Q0 = Q0

        # Set list of valid actions
        self.sa_list = sa_list
        
        # Set initial Q values to Q0, and create set of valid actions
        self.Q = {}
        self.valid_actions = {}

        for (s, a) in self.sa_list:

            if s not in self.Q:
                self.Q[s] = {a : Q0}
            else:
                self.Q[s][a] = Q0

            if s not in self.valid_actions:
                self.valid_actions[s] = set([a])
            else:
                self.valid_actions[s].add(a)
        

    def take_action(self, s, t):

        if self.dither_mode == 'epsilon-greedy':
            
            # Get action corresponding to highest Q
            a = self.get_max_a_Q(s, argmax=True)
            
            anneal_factor = np.exp(- t / self.anneal_timescale)

            if np.random.rand() < anneal_factor * self.dither_param:
                
                # Return random pick from valid actions
                return np.random.choice(list(self.valid_actions[s]))

            else:
                return a

        elif self.dither_mode == 'boltzmann':
            
            anneal_factor = np.exp(- t / self.anneal_timescale)
            
            T = self.dither_param * anneal_factor

            # Get list of valid actions
            valid_actions = list(self.valid_actions[s])

            # Get Q values coorespodning to actions from state s
            Q_ = np.array([self.Q[s][a] for a in valid_actions]) 

            # Calculate Boltzmann probabilities and normalise
            probs = np.exp(Q_ / T)
            probs = probs / probs.sum()
            
            return np.random.choice(valid_actions, p=probs)


    def update_Q(self, s, a, r, s_):
    
        # Get maximum Q corresponding to next state s_
        max_a_Q = self.get_max_a_Q(s_)

        # Apply Q-Learning update rule
        self.Q[s][a] += self.lr * float(r + self.gamma * max_a_Q - self.Q[s][a])


    def get_max_a_Q(self, s, argmax=False):
        '''
            Reuruns the maximum of Q[s] across all valid actions
        '''

        # Get list of valid actions
        valid_actions = list(self.valid_actions[s])

        # Get Q values coorespodning to actions from state s
        Q_ = np.array([self.Q[s][a] for a in valid_actions])

        if argmax:
            # Break ties at random
            a_idx = np.random.choice(np.argwhere(Q_ == np.amax(Q_))[:, 0])
            return valid_actions[a_idx]
        else:
            return np.max(Q_)


    def get_greedy_policy(self):

        pi = {}

        # Get set of states
        states = set([s for (s, a) in self.sa_list])
        
        for s in states:
            # For every state, get action with max Q
            pi[s] = self.get_max_a_Q(s, argmax=True)

        return pi
    
    
    def observe(self, transition):
        t, s, a, r, s_ = transition
        self.add_observations(s, a, r, s_)
        self.last_transition = transition
        
    
    def update_after_step(self, max_buffer_length):
        t, s, a, r, s_ = self.last_transition
        self.update_Q(s, a, r, s_)
        self.last_transition = None
        
        
    def do_before_save(self):
        self.pi = self.get_greedy_policy()
        
        
    def get_name(self):
        
        placeholder = 'QLearningAgent_dither-{}_ditherparam-{}_gamma-{}_lr-{}_Q0-{}-tscale-{}'
        
        name = placeholder.format(self.dither_mode, self.dither_param,
                                  self.gamma, self.lr, self.Q0, self.anneal_timescale)
        return name
        
           
# ============================================================================
# BayesianQAgent class
# ============================================================================


class BayesianQAgent(TabularAgent):
    
    def __init__(self, agent_params):
        
        gamma = agent_params['gamma']
        mu0 = agent_params['mu0']
        lamda = agent_params['lamda']
        alpha = agent_params['alpha']
        beta = agent_params['beta']
        sa_list = agent_params['sa_list']
        T = agent_params['T']
        num_mixture_samples = agent_params['num_mixture_samples']

        super(BayesianQAgent, self).__init__(gamma, T)

        # Dict for holding posterior phyperparameters
        self.Qpost = {}
        
        # Number of mixture samples for update
        self.num_mixture_samples = num_mixture_samples
        
        # Prior parameters
        self.mu0 = agent_params['mu0']
        self.lamda = agent_params['lamda']
        self.alpha = agent_params['alpha']
        self.beta = agent_params['beta']
        self.sa_list = agent_params['sa_list']
        

        # Set normal-gamma prior parameters for each state-action
        for s, a in sa_list:

            if s not in self.Qpost:
                self.Qpost[s] = {}

            self.Qpost[s][a] = (mu0, lamda, alpha, beta)


    def take_action(self, s, t, reduce_max=True):

        # Sample q values for each action from current state
        qs, acts = self.sample_q(s)

        if reduce_max:
            # Return action corresponding to maximum q
            return acts[np.argmax(qs)]
        else:
            return qs, acts


    def sample_q(self, s):

        # Arrays for holding q samples and corresponding actions
        qs, acts = [], []

        for a, hyp in self.Qpost[s].items():

            # Sample from student-t distribution
            st = np.random.standard_t(2 * hyp[2])
            
            # q sample from t:  m0 + t * (beta / (lamda * alpha))**0.5
            # qs.append(hyp[0] + st * (hyp[3] / (hyp[1] * hyp[2]))**0.5)
            qs.append(hyp[0] + st * (hyp[3] / (hyp[1] * (hyp[2] - 1)))**0.5)
            acts.append(a)
            
        return np.array(qs), np.array(acts)


    def kl_matched_hyps(self, s, a, r, s_, done=False):

        # If episode finised at s_, R_t is 0, otherwise R_t will be overwritten
        R_t = np.zeros((self.num_mixture_samples,))

        if not done:

            # Find the action from s_ with the largest mean
            a_ = self.max_mu0_action(s_)

            # Sample R_t
            mu0_t, lamda_t, alpha_t, beta_t = self.Qpost[s_][a_]
            st = np.random.standard_t(2 * alpha_t, size=(self.num_mixture_samples,))
            coeff = (beta_t * (lamda_t + 1) / (alpha_t * lamda_t))**0.5
            R_t = mu0_t + st * coeff

        R_t = r + self.gamma * R_t
    
        # R_sa posterior hyperparameters
        mu0_sa, lamda_sa, alpha_sa, beta_sa = self.Qpost[s][a]
        
        # R_sa posterior hyperparameters updated for each sample
        mu0_ = (lamda_sa * mu0_sa + R_t) / (lamda_sa + 1)
        lamda_ = np.tile(lamda_sa + 1, reps=(mu0_.shape[0],))
        alpha_ = np.tile(alpha_sa + 0.5, reps=(mu0_.shape[0],))
        beta_ = beta_sa + lamda_sa * 0.5 * (R_t - mu0_sa)**2 / (lamda_sa + 1)

        # Sample mu and tau for each set of updated hyperparameters
        mus, taus = normal_gamma(mu0_, lamda_, alpha_, beta_)

        # MC estimates of moments
        E_tau = np.mean(taus)
        E_mu_tau = np.mean(mus * taus)
        E_mu2_tau = np.mean(mus**2 * taus)
        E_log_tau = np.mean(np.log(taus))

        # phi(x) = f^-1(x) where f(x) = log(x) - digamma(x)
        phi_term = phi(np.log(E_tau) - E_log_tau)

        # Calculate hyperparameters of KL-matched normal gamma
        m0_ = E_mu_tau / E_tau
        lamda_ = 1 / (1e-12 + E_mu2_tau - E_tau * m0_**2)
        alpha_ = max(1e-6, phi_term)
        beta_ = alpha_ / E_tau

        return m0_, lamda_, alpha_, beta_


    def max_mu0_action(self, s):
        
        # Get actions and corresponding hyperparameters of R_sa distribution
        a_mu0 = [(a, hyp[0]) for (a, hyp) in self.Qpost[s].items()]
        a, mu0 = [np.array(arr) for arr in zip(*a_mu0)]

        return a[np.argmax(mu0)]
    

    def observe(self, transition):
        t, s, a, r, s_ = transition
        self.add_observations(s, a, r, s_)
        self.last_transition = transition
        
    
    def update_after_step(self, max_buffer_length):
        t, s, a, r, s_ = self.last_transition
        hyps = self.kl_matched_hyps(s, a, r, s_, done=False)
        self.Qpost[s][a] = hyps
        self.last_transition = None
        
        
    def do_before_save(self):
        pass
        
        
    def get_name(self):
        
        placeholder = 'BayesianQAgent_gamma-{}_mu0-{}_lamda-{}_alpha-{}_beta-{}'
        
        name = placeholder.format(self.gamma, self.mu0, self.lamda,
                                  self.alpha, self.beta)
        return name
    
    
    def get_greedy_policy(self):
        
        num_s = len(set([s for (s, a) in self.sa_list]))
        
        # Empty policy
        pi = np.zeros((num_s,))
        
        for (s, a) in self.sa_list:
            
            qs, acts = [], []
            
            for a_, hyp in self.Qpost[s].items():

                qs.append(hyp[0])
                acts.append(a)
               
            pi[s] = acts[np.argmax(qs)]
            
        return pi
        
        
        
# ============================================================================
# PSRLAgent agent definition
# ============================================================================


class PSRLAgent(TabularAgent):

    def __init__(self, params):
        
        gamma = params['gamma']
        T = params['T']
        kappa = params['kappa']
        mu0 = params['mu0']
        lamda = params['lamda']
        alpha = params['alpha']
        beta = params['beta']
        sa_list = params['sa_list']
        num_pi_iter = params['num_pi_iter']
        num_dyn_samples = params['num_dyn_samples']

        super(PSRLAgent, self).__init__(gamma, T)

        self.kappa = kappa
        self.mu0 = mu0
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.sa_list = sa_list
        self.num_dyn_samples = num_dyn_samples
        
        # Initialise dictionaries of posterior hyperparameters
        self.Ppost = {}
        self.Rpost = {}

        # Count the number of states
        self.num_s = len(set([s for (s, a) in sa_list]))
        
        # Count the number of actions
        self.num_a = len(set([a for (s, a) in sa_list]))
        
        # Buffer for storing current episode
        self.episode_buffer = []
        
        # Number of policy iteration steps
        self.num_pi_iter = num_pi_iter

        # For all sa pairs, initialise posterior hyps to prior
        for s in range(self.num_s):
            for a in range(self.num_a):
                self.Ppost[(s, a)] = kappa * np.ones((self.num_s,))
                
        # For valid sa pairs, initialise posterior hyps to prior
        for s, a in sa_list:
            for s_ in range(self.num_s):
                self.Rpost[(s, a, s_)] = (mu0, lamda, alpha, beta)
                
        if len(sa_list) > 0: self.sample_posterior_and_update_continuing_policy()


    def sample_posterior(self):

        # Initialise posterior arrays (dynamics 0, reward large negative)
        P_ = np.zeros((self.num_s, self.num_a, self.num_s))
        R_ = -1e6 * np.ones((self.num_s, self.num_a, self.num_s))

        for (s, a), kappas in self.Ppost.items():
            P_[s, a, :] = np.random.dirichlet(kappas)
            
        for (s, a, s_), hyps in self.Rpost.items():
            mu0, lamda, alpha, beta = hyps
            R_[s, a, s_] = normal_gamma(mu0, lamda, alpha, beta)[0]

        return P_, R_


    def update_posterior(self):

        # Transition counts and reward sums
        p_counts = np.zeros((self.num_s, self.num_a, self.num_s))
        r_sums = np.zeros((self.num_s, self.num_a, self.num_s))
        r_counts = np.zeros((self.num_s, self.num_a, self.num_s))

        for (s, a, r, s_) in self.episode_buffer:
            p_counts[s, a, s_] += 1
            r_sums[s, a, s_] += r
            r_counts[s, a, s_] += 1

        # Update dynamics posterior
        for (s, a), kappas in self.Ppost.items():
            # Dirichlet posterior params are prior params plus counts
            self.Ppost[(s, a)] = self.Ppost[(s, a)] + p_counts[s, a]

        # Update rewards posterior
        for (s, a, s_), (mu0, lamda, alpha, beta) in self.Rpost.items():

            # Calculate moments
            M1 = r_sums[s, a, s_] / max(1, r_counts[s, a, s_])
            M2 = r_sums[s, a, s_]**2 / max(1, r_counts[s, a, s_])
            n = r_counts[s, a, s_]
            
            mu0_ = (lamda * mu0 + n * M1) / (lamda + n)
            lamda_ = lamda + n
            alpha_ = alpha + 0.5 * n
            beta_ = beta + 0.5 * n * (M2 - M1**2)
            beta_ = beta_ + n * lamda * (M1 - mu0)**2 / (2 * (lamda + n))    

            # Update dictionary
            self.Rpost[(s, a, s_)] = (mu0_, lamda_, alpha_, beta_)

        # Reset episode buffer
        self.episode_buffer = []


    def take_action(self, s, t):
            
        if self.T == float('inf'):
            return self.pi[s]
        else:
            return self.pi[t, s]


    def update_episode_buffer(self, s, a, r, s_):
        self.episode_buffer.append([s, a, r, s_])
        
        
    def observe(self, transition):
        t, s, a, r, s_ = transition
        self.add_observations(s, a, r, s_)
        self.episode_buffer.append([s, a, r, s_])
    
    
    def update_after_step(self, max_buffer_length):
        if len(self.episode_buffer) >= max_buffer_length:
            self.update_posterior()
            self.sample_posterior_and_update_continuing_policy()
    
    
    def update_after_episode(self):
        if len(self.episode_buffer) > 0:
            self.update_posterior()
        
        
    def sample_posterior_and_update_continuing_policy(self):
    
        # Sample dynamics and rewards posterior
        P, R = self.sample_posterior()
        
        # Solve Bellman equation by policy iteration
        pi, Q = solve_tabular_continuing_PI(P, R, self.gamma, self.num_pi_iter)
        
        self.pi = pi
        
        
    def do_before_save(self):
        pass
    
        
    def get_name(self):
        
        placeholder = 'PSRLAgent_gamma-{}_kappa-{}_mu0-{}_lamda-{}_alpha-{}_beta-{}'
        
        name = placeholder.format(self.gamma, self.kappa, self.mu0,
                                  self.lamda, self.alpha, self.beta)
        return name
    
    
    def get_greedy_policy(self):
        
        Q = np.zeros((self.num_dyn_samples, self.num_s, self.num_a))
        
        for i in range(5):
            # Sample dynamics and rewards posterior
            P_, R_ = self.sample_posterior()

            # Solve Bellman equation by policy iteration
            pi, Q_ = solve_tabular_continuing_PI(P_,
                                                 R_,
                                                 self.gamma,
                                                 self.num_pi_iter)
            
            Q[i] = Q_
            
        return Q.mean(axis=0).argmax(axis=-1)
            
    

# ============================================================================
# MomentMatching agent definition
# ============================================================================


class MomentMatchingAgent(TabularAgent):

    def __init__(self, params):
        
        gamma = params['gamma']
        kappa = params['kappa']
        mu0 = params['mu0']
        lamda = params['lamda']
        alpha = params['alpha']
        beta = params['beta']
        T = params['T']
        
        self.kappa = params['kappa']
        self.mu0 = params['mu0']
        self.lamda = params['lamda']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.sa_list = params['sa_list']
        self.zeta = params['zeta']
        self.num_PI_steps = params['num_PI_steps']
        self.num_dyn_samples = params['num_dyn_samples']
        
        self.thompson = params['thompson'] if ('thompson' in params) else False
        
        super(MomentMatchingAgent, self).__init__(gamma, T)

        # Initialise dictionaries of posterior hyperparameters
        self.Ppost = {}
        self.Rpost = {}

        # Count the number of states
        self.num_s = len(set([s for (s, a) in self.sa_list])) 
        
        # Count the number of actions
        self.num_a = len(set([a for (s, a) in self.sa_list]))
        
        # Buffer for storing current episode
        self.episode_buffer = []

        # For all sa pairs, initialise posterior hyps to prior
        for s in range(self.num_s):
            for a in range(self.num_a):
                self.Ppost[(s, a)] = kappa * np.ones((self.num_s,))
                
        # For valid sa pairs, initialise posterior hyps to prior
        for s in range(self.num_s):
            for a in range(self.num_a):
                for s_ in range(self.num_s):
                    if (s, a) in self.sa_list:
                        self.Rpost[(s, a, s_)] = (mu0, lamda, alpha, beta)
                    else:
                        self.Rpost[(s, a, s_)] = (-1e12, 1., 1e6, 1e6)
                        
        if len(self.sa_list) > 0:
            self.do_before_save()


    def update_posterior(self):

        # Transition counts and reward sums
        p_counts = np.zeros((self.num_s, self.num_a, self.num_s))
        r_sums = np.zeros((self.num_s, self.num_a, self.num_s))
        r_counts = np.zeros((self.num_s, self.num_a, self.num_s))

        for (s, a, r, s_) in self.episode_buffer:
            p_counts[s, a, s_] += 1
            r_sums[s, a, s_] += r
            r_counts[s, a, s_] += 1

        # Update dynamics posterior
        for (s, a), kappas in self.Ppost.items():
            # Dirichlet posterior params are prior params plus counts
            self.Ppost[(s, a)] = self.Ppost[(s, a)] + p_counts[s, a]

        # Update rewards posterior
        for (s, a, s_), (mu0, lamda, alpha, beta) in self.Rpost.items():

            # Calculate moments
            M1 = r_sums[s, a, s_] / max(1, r_counts[s, a, s_])
            M2 = r_sums[s, a, s_]**2 / max(1, r_counts[s, a, s_])
            n = r_counts[s, a, s_]
            
            mu0_ = (lamda * mu0 + n * M1) / (lamda + n)
            lamda_ = lamda + n
            alpha_ = alpha + 0.5 * n
            beta_ = beta + 0.5 * n * (M2 - M1**2)
            beta_ = beta_ + n * lamda * (M1 - mu0)**2 / (2 * (lamda + n))    

            # Update dictionary
            self.Rpost[(s, a, s_)] = (mu0_, lamda_, alpha_, beta_)

        # Reset episode buffer
        self.episode_buffer = []


    def take_action(self, s, t, reduce_max=True):

        if self.T == float('inf'):
            
            if self.thompson:
                mean = self.mu_z_sa[s, :]
                std = (self.val_var_val[s, :] / self.gamma**2) ** 0.5
                q_samples = np.random.normal(loc=mean, scale=self.zeta*std)
                if reduce_max:
                    return np.argmax(q_samples)
                else:
                    return q_samples, None
                                 
            else:
                return self.pi[s]

    
    def ucb_policy_iteration(self, each_term=False, zeta=None):

        zeta = self.zeta if zeta is None else zeta
        
        # Initialise policy
        pi = self.starting_policy()
        
        # Sample dynamics models
        P = self.sample_dynamics()

        # Get arrays of reward posterior parameters (num_s, num_a, num_s)
        rew_params = self.reward_posterior_arrays()

        for i in range(self.num_PI_steps):

            # Solve for mean values of z
            mu_z = self.mu_z(P, pi, rew_params)

            # Solve for variances
            var_z = self.var_z(P, pi, mu_z, rew_params)

            # Get backed-up mean
            mu_z_sa = self.backup_mu_z(P, mu_z, rew_params)

            # Get backed-up variance
            var_z_sa, var_z_sa_terms = self.backup_var_z(P, mu_z, var_z, rew_params)
            
            # Calculate UCB heuristic
            ucb = mu_z_sa + zeta * var_z_sa**0.5

            # Improve policy
            pi = np.array([np.random.choice(np.argwhere(mu_z_s == np.amax(mu_z_s))[0]) for mu_z_s in mu_z_sa])
            
        if each_term:
            return pi, mu_z_sa, var_z_sa_terms
        else:
            return pi
        
        
    def greedy_policy_iteration_with_variance_output(self):

        # Initialise policy
        pi = self.starting_policy()
        
        # Sample dynamics models
        P = self.sample_dynamics()

        # Get arrays of reward posterior parameters (num_s, num_a, num_s)
        rew_params = self.reward_posterior_arrays()
        
        # Values to return
        mu_z_sa, mu_z = None, None

        for i in range(self.num_PI_steps):

            # Solve for mean values of z
            mu_z = self.mu_z(P, pi, rew_params)

            # Get backed-up mean
            mu_z_sa = self.backup_mu_z(P, mu_z, rew_params)

            # Improve policy
            pi = np.array([np.random.choice(np.argwhere(mu_z_s == np.amax(mu_z_s))[0]) for mu_z_s in mu_z_sa])
            
        # Solve for variances
        var_z = self.var_z(P, pi, mu_z, rew_params)

        # Get backed-up mean
        mu_z_sa = self.backup_mu_z(P, mu_z, rew_params)

        # Get backed-up variance
        var_z_sa, var_z_sa_terms = self.backup_var_z(P, mu_z, var_z, rew_params)
            
        return pi, mu_z_sa, var_z_sa_terms
        


    def mu_z(self, P, pi, rew_params):
        '''
            Return mean of z posterior    
        '''

        idx = np.arange(pi.shape[0])

        # Unpack reward parameters
        mu0, _, _, _ = rew_params 

        # Get mean reward (num_s,)
        P_mu0 = np.mean(np.einsum('nijk, ijk -> nij', P, mu0), axis=0)[idx, pi]
        
        # Get mean transition matrix (num_s, num_s)
        P_ = np.mean(P[:, idx, pi, :], axis=0)

        # Solve for mean values of z
        A = np.eye(P_.shape[0]) - self.gamma * P_ 
        mu_z = np.linalg.solve(A, P_mu0)

        return mu_z


    def backup_mu_z(self, P, mu_z, rew_params):
        '''
            Return mean of z posterior by Bellman backup
        '''

        # Unpack reward parameters
        mu0, lamda, alpha, beta = rew_params 

        # Get mean reward (num_s,)
        P_mu0 = np.mean(np.einsum('nijk, ijk -> nij', P, mu0), axis=0)

        # Get the mean transition matrix
        P_ = np.mean(P, axis=0)

        return P_mu0 + self.gamma * np.einsum('ijk, k -> ij', P_, mu_z) 


    def var_z(self, P, pi, mu_z, rew_params, show=False):
        '''
            Returns variance of z posterior
        '''

        idx = np.arange(pi.shape[0])

        # Unpack reward parameters
        mu0, lamda, alpha, beta = rew_params

        # Variance of rewards due to uncertainty in dynamics
        var_rew_dyn = np.einsum('nijk, ijk -> nij', P, mu0).var(axis=0)
        
        if show: print('var_rew_dyn\n', var_rew_dyn)

        # Variance of rewards due to uncertainty of reward mean
        student_var = beta / ((alpha - 1) * lamda)
        temp_rew_rew = P * student_var[None, :, :, :]
        var_rew_rew = np.mean(temp_rew_rew, axis=0).sum(axis=-1)
        
        if show: print('var_rew_rew\n', var_rew_rew)

        # Reward value covariance due to uncertainty of dynamics
        mu_r_dyn = np.einsum('nijk, ijk -> nij', P, mu0)
        mu_z_dyn = np.einsum('nijk, k -> nij', P, mu_z)
        cov = (mu_r_dyn * mu_z_dyn).mean(axis=0) - \
                mu_r_dyn.mean(axis=0) * mu_z_dyn.mean(axis=0)
        
        if show: print('cov\n', cov)

        # Value variance due to dynamics uncertainty
        var_z_dyn = np.einsum('nijk, k -> nij', P, mu_z).var(axis=0)
        
        if show: print('var_z_dyn\n', var_z_dyn)

        # Get mean transition matrix (num_s, num_s)
        P_ = np.mean(P, axis=0)[idx, pi, :]

        # Calculate total variance
        total_var = (var_rew_dyn + var_rew_rew + \
                2 * self.gamma * cov + self.gamma**2 * var_z_dyn)[idx, pi]

        A = np.eye(P_.shape[0]) - self.gamma**2 * P_
        var_z = np.linalg.solve(A, total_var)
        
        if show:
            print('var_z\n', var_z)
            print(np.linalg.det(A), np.linalg.slogdet(A))
            
            return var_rew_dyn, var_rew_rew, cov, var_z_dyn, total_var, P_
        
        return var_z


    def backup_var_z(self, P, mu_z, var_z, rew_params):
        '''
            Returns z vars. If pi != None, returns vars for that policy.
            If pi == None, var_z must not be None. In this case the function
            returns the var_z(s, a) values.
        '''

        # Unpack reward parameters
        mu0, lamda, alpha, beta = rew_params

        # Variance of rewards due to uncertainty in dynamics
        var_rew_dyn = np.einsum('nijk, ijk -> nij', P, mu0).var(axis=0)

        # Variance of rewards due to uncertainty of reward mean
        student_var = beta / ((alpha - 1) * lamda)
        temp_rew_rew = P * student_var[None, :, :, :]
        var_rew_rew = np.mean(temp_rew_rew, axis=0).sum(axis=-1)

        # Reward value covariance due to uncertainty of dynamics
        mu_r_dyn = np.einsum('nijk, ijk -> nij', P, mu0)
        mu_z_dyn = np.einsum('nijk, k -> nij', P, mu_z)
        cov = (mu_r_dyn * mu_z_dyn).mean(axis=0) - \
                mu_r_dyn.mean(axis=0) * mu_z_dyn.mean(axis=0)

        # Value variance due to dynamics uncertainty
        var_z_dyn = np.einsum('nijk, k -> nij', P, mu_z).var(axis=0)

        # Get mean transition matrix (num_s, num_s)
        P_ = np.mean(P, axis=0)

        # Calculate total variance
        total_var = (var_rew_dyn + var_rew_rew + \
                     2 * self.gamma * cov + self.gamma**2 * var_z_dyn)

        var_z = np.einsum('nijk, k -> nij', P, var_z).mean(axis=0)

        terms = (var_rew_dyn, var_rew_rew, 2 * self.gamma * cov,
                 self.gamma**2 * var_z_dyn, self.gamma**2 * var_z)
        
        return total_var + self.gamma**2 * var_z, terms


    def reward_posterior_arrays(self):
        
        num_s, num_a = self.num_s, self.num_a
        m0, lamda, alpha, beta = [np.zeros(shape=(num_s, num_a, num_s)) \
                                  for i in range(4)]

        for ((s, a, s_), p) in self.Rpost.items():
            m0[s, a, s_], lamda[s, a, s_], alpha[s, a, s_], beta[s, a, s_] = p
        
        return m0, lamda, alpha, beta

    
    def starting_policy(self):
        '''
            Initialises starting policy from allowed state-action pairs
        '''
        
        # Set policy to zeros
        pi = np.zeros(shape=(self.num_s,), dtype=np.int)

        # Set policy to an action from the allowed actions
        for (s, a) in self.sa_list:
            pi[s] = int(a)
        
        return pi 

    
    def sample_dynamics(self):
        
        # Initialise array of samples
        dyns = []

        # Loop over all states
        for s in range(self.num_s):

            dyns.append([])

            # Loop over all actions
            for a in range(self.num_a):
                
                # Dirichlet distribution parameters from posterior
                kappas = self.Ppost[(s, a)]

                # Sample transitions from current state-action pair
                samples = np.random.dirichlet(kappas, size=self.num_dyn_samples)

                # Store samples
                dyns[-1].append(samples.T)

        # Rearrange into (num_samples, num_s, num_a, num_s)
        dyns = np.rollaxis(np.array(dyns), 3)
        
        return dyns


    def update_episode_buffer(self, s, a, r, s_):
        self.episode_buffer.append([s, a, r, s_])
        
        
    def observe(self, transition):
        t, s, a, r, s_ = transition
        self.add_observations(s, a, r, s_)
        self.episode_buffer.append([s, a, r, s_])
    
    
    def update_after_step(self, max_buffer_length):
        
        if len(self.episode_buffer) >= max_buffer_length:
            
            self.update_posterior()
            if self.thompson:
                pi, mu_z_sa, var_z_sa_terms = self.greedy_policy_iteration_with_variance_output()
                self.pi, self.mu_z_sa, self.var_z_sa_terms = pi, mu_z_sa, var_z_sa_terms
            else:
                self.pi = self.ucb_policy_iteration()
    
    
    def update_after_episode(self):
        if len(self.episode_buffer) > 0:
            self.update_posterior()
            
            
    def do_before_save(self):
        
        pi, mu_z_sa, var_terms = (None,) * 3
        
        if self.thompson:
            pi, mu_z_sa, var_terms = self.greedy_policy_iteration_with_variance_output()
        else:
            pi, mu_z_sa, var_terms = self.ucb_policy_iteration(each_term=True)
        
        rew_var_dyn, rew_var_rew, cov_rew_val, val_var_dyn, val_var_val = var_terms
        
        self.pi = pi
        self.mu_z_sa = mu_z_sa
        self.rew_var_dyn = rew_var_dyn
        self.rew_var_rew = rew_var_rew
        self.cov_rew_val = cov_rew_val
        self.val_var_dyn = val_var_dyn
        self.val_var_val = val_var_val
            
            
    def get_name(self):
        
        placeholder = 'MomentMatchingAgent_gamma-{}_kappa-{}_mu0-{}_lamda-{}_alpha-{}_beta-{}_zeta-{}'
        
        if self.thompson:
            placeholder = placeholder + '-thompson'
        
        name = placeholder.format(self.gamma, self.kappa, self.mu0,
                                  self.lamda, self.alpha, self.beta, self.zeta)
        
        return name
    
    
    def get_greedy_policy(self):
        return self.ucb_policy_iteration(each_term=False, zeta=None)
        
        
        
# ============================================================================
# UbeNoUnrollAgent class
# ============================================================================

            
class UbeNoUnrollAgent(TabularAgent):

    def __init__(self, params):
        
        gamma = params['gamma']
        Rmax = params['Rmax']
        kappa = params['kappa']
        mu0 = params['mu0']
        lamda = params['lamda']
        alpha = params['alpha']
        beta = params['beta']
        zeta = params['zeta']
        T = params['T']
        sa_list = params['sa_list']
        num_pi_iter = params['num_pi_iter']
        num_dyn_samples = params['num_dyn_samples']

        super(UbeNoUnrollAgent, self).__init__(gamma, T)

        # Set constants
        self.gamma = gamma
        self.T = T
        self.Rmax = Rmax
        self.zeta = zeta
        self.kappa = kappa
        self.mu0 = mu0
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.sa_list = sa_list
        self.num_pi_iter = num_pi_iter
        self.num_dyn_samples = num_dyn_samples
        self.num_s = len(set([s for (s, a) in self.sa_list])) 
        self.num_a = len(set([a for (s, a) in self.sa_list]))
 
        # Set episode buffer
        self.episode_buffer = []

        # Dynamics and rewards posterior
        self.Ppost = {}
        self.Rpost = {}

        # Init. dynamics posterior - unroll MDP (assumption 1 in paper)
        for s in range(self.num_s):
            for a in range(self.num_a):
                self.Ppost[(s, a)] = kappa * np.ones((self.num_s,))
                
        # Init. rewards posteriror - unroll MDP
        for s in range(self.num_s):
            for a in range(self.num_a):
                for s_ in range(self.num_s):
                    if (s, a) in sa_list:
                        self.Rpost[(s, a, s_)] = (mu0, lamda, alpha, beta)
                    else:
                        self.Rpost[(s, a, s_)] = (-1e6, 1., 2., 1e6)
                        
        if len(sa_list) > 0: self.set_Q_posterior()


    def update_episode_buffer(self, frame):
        self.episode_buffer.append(frame)


    def update_posterior(self):

        # Transition counts and reward sums
        p_counts = np.zeros((self.num_s, self.num_a, self.num_s))
        r_sums = np.zeros((self.num_s, self.num_a, self.num_s))
        r_counts = np.zeros((self.num_s, self.num_a, self.num_s))

        for (s, a, r, s_) in self.episode_buffer:
            p_counts[s, a, s_] += 1
            r_sums[s, a, s_] += r
            r_counts[s, a, s_] += 1

        # Update dynamics posterior
        for (s, a), kappas in self.Ppost.items():
            # Dirichlet posterior params are prior params plus counts
            self.Ppost[(s, a)] = self.Ppost[(s, a)] + p_counts[s, a]

        # Update rewards posterior
        for (s, a, s_), (mu0, lamda, alpha, beta) in self.Rpost.items():

            # Calculate moments
            M1 = r_sums[s, a, s_] / max(1, r_counts[s, a, s_])
            M2 = r_sums[s, a, s_]**2 / max(1, r_counts[s, a, s_])
            n = r_counts[s, a, s_]
            
            mu0_ = (lamda * mu0 + n * M1) / (lamda + n)
            lamda_ = lamda + n
            alpha_ = alpha + 0.5 * n
            beta_ = beta + 0.5 * n * (M2 - M1**2)
            beta_ = beta_ + n * lamda * (M1 - mu0)**2 / (2 * (lamda + n))    

            # Update dictionary
            self.Rpost[(s, a, s_)] = (mu0_, lamda_, alpha_, beta_)

        # Reset episode buffer
        self.episode_buffer = []


    def set_Q_posterior(self):
        '''
            Computes the approximation (diagonal gaussian) of the Q posterior
            under policy pi.
        '''
        
        # Compute the greedy policy and corresponding Q values
        pi, Qmu = self.policy_iteration()

        # Compute the uncertainty (variance) of Q
        Qvar = self.solve_unrolled_bellman(self.local_rew_var,
                                           self.gamma**2,
                                           pi)

        self.Qmu = Qmu
        self.Qvar = Qvar
        self.pi = pi

    
    def thompson_action(self, s, t, reduce_max=True):

        # Posterior mean and variance
        mu = self.Qmu[s, :]
        var = self.Qvar[s, :]

        # Sample Q from diagonal gaussian
        Q_sample = np.random.normal(loc=mu, scale=(self.zeta * var**0.5))
        
        if reduce_max:
            return np.argmax(Q_sample)
        else:
            return Q_sample, None
    
    
    def take_action(self, s, t):

        if self.T == float('inf'):
            return self.thompson_action(s, t)


    def policy_iteration(self):

        pi = np.zeros((self.num_s,), dtype=np.int)

        for i in range(self.num_pi_iter):

            # Policy evaluation step
            Q = self.solve_unrolled_bellman(self.local_rew_mean,
                                            self.gamma,
                                            pi)
            
            # Policy improvement step
            pi = np.argmax(Q, axis=-1)

        return pi, Q


    def solve_unrolled_bellman(self, local, discount, pi):
        '''
            Solves Bellman equation for unrolled MDP of horizon H. Works for
            both BE and UBE. local must be a function returning the mean
            reward (BE) or the posterior uncertainty (UBE) at (t, s, a, s).
            Dicount must be gamma (BE) or gamma^2 (UBE). pi is the policy
            and must be time-indexable (t, s).
        '''
        
        s_idx = np.arange(self.num_s)
        P = self.sample_dynamics()
        
        loc = local(P)

        # The Bellman values we are solving for 
        backed = np.zeros((self.num_s, self.num_a))

        for t in range(self.num_pi_iter):
            
            # Computed backed up values from one step forward in time
            pi_backed = backed[s_idx, pi]
            P_pi_backed = np.einsum('nijk, k -> nij', P, pi_backed)
            P_pi_backed = P_pi_backed.mean(axis=0)

            backed = loc + discount * P_pi_backed

        return backed
    

    def local_rew_mean(self, P):

        means = [[[self.Rpost[(s, a, s_)][0] for s_ in range(self.num_s)]\
                                             for a in range(self.num_a)] \
                                             for s in range(self.num_s)]
        means = np.array(means)

        return np.einsum('nijk, ijk -> nij', P, means).mean(axis=0)


    def local_rew_var(self, P, each_term=False):

        # Posterior predictive of rewards (gaussian and NG) is student-t
        lamdas = [[[self.Rpost[(s, a, s_)][1] for s_ in range(self.num_s)]\
                                              for a in range(self.num_a)] \
                                              for s in range(self.num_s)]

        alphas = [[[self.Rpost[(s, a, s_)][2] for s_ in range(self.num_s)]\
                                              for a in range(self.num_a)]\
                                              for s in range(self.num_s)]
        alphas = np.array(alphas)
        mu_var = 2 * alphas / ((alphas - 1) * lamdas)

        # Mean of epistemic variance of rewards
        mean_mu_var = np.einsum('nijk, ijk -> nij', P, mu_var).mean(axis=0)

        # Variance of reward mean due to epistemic dynamics uncertainty
        mu_mu = [[[self.Rpost[(s, a, s_)][0] for s_ in range(self.num_s)]\
                                             for a in range(self.num_a)] \
                                             for s in range(self.num_s)]
        mu_mu = np.array(mu_mu)
 
        var_mu_mean = np.einsum('nijk, ijk -> nij', P, mu_mu).var(axis=0)

        # Total uncertainty is sum of two sources of epistemic uncertainty
        var_rew = mean_mu_var + var_mu_mean

        # Calculate second term of local uncertainty
        Qmax2 = (self.Rmax / (1 - self.gamma))**2
        Qmax_term = (Qmax2 * P.var(axis=0) / P.mean(axis=0)).sum(axis=-1)

        # Local uncertainty is the sum of these two terms
        if each_term:
            return var_rew, Qmax_term
        else:
            return var_rew + Qmax_term


    def sample_dynamics(self):

        dyns = []
        for s in range(self.num_s):
            dyns.append([])
            for a in range(self.num_a):

                # Dirichlet distribution parameters from posterior
                kappas = self.Ppost[(s, a)]

                # Sample transitions from current state-action pair
                samples = np.random.dirichlet(kappas, size=self.num_dyn_samples)

                # Store samples
                dyns[-1].append(samples.T)

        # Rearrange into (num_samples, num_s, num_a, num_s)
        dyns = np.rollaxis(np.array(dyns), 3)

        return dyns
    
    
    def observe(self, transition):
        t, s, a, r, s_ = transition
        self.add_observations(s, a, r, s_)
        self.episode_buffer.append([s, a, r, s_])
    
    
    def update_after_step(self, max_buffer_length):
        if len(self.episode_buffer) >= max_buffer_length:
            self.update_posterior()
            self.set_Q_posterior()
            
    
    def update_after_episode(self):
        if len(self.episode_buffer) > 0:
            self.update_posterior()
            
            
    def do_before_save(self):
        
        # Sets policy, mean and var Q
        self.set_Q_posterior()
        
        P = self.sample_dynamics()
        var_rew, var_Qmax = self.local_rew_var(P, each_term=True)
        
        self.local_var_rew = var_rew
        self.var_Qmax = var_Qmax
            
            
    def get_name(self):
        placeholder = 'UbeNoUnrollAgent_gamma-{}_kappa-{}_mu0-{}_lamda-{}_alpha-{}_beta-{}_zeta-{}'
        
        name = placeholder.format(self.gamma, self.kappa, self.mu0,
                                  self.lamda, self.alpha, self.beta, self.zeta)
        
        return name
    
    
    def get_greedy_policy(self):
        
        # Sets policy, mean and var Q
        self.set_Q_posterior()
        
        return self.pi
        
            