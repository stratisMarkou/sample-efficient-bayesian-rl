import numpy as np

import pickle
import os
from copy import deepcopy

from scipy.special import digamma
from pynverse import inversefunc

from utils import bql_f_inv,                    \
                  normal_gamma,                 \
                  solve_tabular_continuing_PI


# ============================================================================
# General Tabular agent class
# ============================================================================


class TabularAgent:
    
    def __init__(self, gamma):

        # Discount factor
        self.gamma = gamma
        
        
    def add_observations(self, s, a, r, s_):
        """ Add observations to log. """

        s, a, r, s_ = [np.array([data]) for data in [s, a, r, s_]]
        
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
        
              
    def save_copy(self, location, name):
        """ Save a copy of the agent. """ 

        fhandle = open(location + '/' + name, 'wb')
        pickle.dump(self, fhandle)
        fhandle.close()
        
        
# ============================================================================
# QLearningAgent class
# ============================================================================


class QLearningAgent(TabularAgent):
    
    def __init__(self, params):

        # Set QLearning agent parameters
        self.gamma = params['gamma']
        self.lr = params['lr']
        self.sa_list = params['sa_list']
        self.Q0 = params['Q0']
        self.dither_mode = params['dither_mode']
        self.dither_param = params['dither_param']
        self.anneal_timescale = params['anneal_timescale']

        # Array for storing previous Q posterior
        self.Qlog = []

        super(QLearningAgent, self).__init__(self.gamma)
        
        # Set initial Q values to Q0, and create set of valid actions
        self.Q = {}
        self.valid_actions = {}

        # List of valid state-actions
        for (s, a) in self.sa_list:

            if s not in self.Q:
                self.Q[s] = {a : self.Q0}
            else:
                self.Q[s][a] = self.Q0

            if s not in self.valid_actions:
                self.valid_actions[s] = set([a])
            else:
                self.valid_actions[s].add(a)
        

    def take_action(self, s, t):
        """ Take epsilon-greedy or boltzmann action. """

        # Compute annealing factor for epsilon or T
        anneal_factor = np.exp(- t / self.anneal_timescale)

        if self.dither_mode == 'epsilon-greedy':
            
            # Get action corresponding to highest Q
            a = self.get_max_a_Q(s, argmax=True)
            
            if np.random.rand() < anneal_factor * self.dither_param:
                
                # Return random pick from valid actions
                return np.random.choice(list(self.valid_actions[s]))

            else:
                return a

        elif self.dither_mode == 'boltzmann':
            
            # Get list of valid actions from state s
            valid_actions = list(self.valid_actions[s])

            # Get Q values coorespodning to actions from state s
            Q_ = np.array([self.Q[s][a] for a in valid_actions]) 

            # Calculate Boltzmann probabilities and normalise
            probs = np.exp(Q_ / (self.dither_param * anneal_factor))
            probs = probs / probs.sum()
            
            return np.random.choice(valid_actions, p=probs)


    def update_Q(self, s, a, r, s_):
        """ Update Q-estimates using Temporal Differences update. """
        
        # Get maximum Q corresponding to next state s_
        max_a_Q = self.get_max_a_Q(s_)

        # Apply Q-Learning update rule
        self.Q[s][a] += self.lr * (r + self.gamma * max_a_Q - self.Q[s][a])


    def get_max_a_Q(self, s, argmax=False):
        """ Returns the maximum of Q[s] across all valid actions. """

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

    
    def observe(self, transition):
        t, s, a, r, s_ = transition
        self.add_observations(s, a, r, s_)
        self.last_transition = transition
        
    
    def update_after_step(self, max_buffer_length, log):

        # Log Q values
        if log: self.Qlog.append(deepcopy(self.Q))

        # Update Q values
        t, s, a, r, s_ = self.last_transition
        self.update_Q(s, a, r, s_)
        self.last_transition = None
        
               
    def get_name(self):
        
        name = 'QLearningAgent_{}_param-{}_gamma-{}_lr-{}_Q0-{}-tscale-{}'
        
        name = name.format(self.dither_mode,
                           self.dither_param,
                           self.gamma,
                           self.lr,
                           self.Q0,
                           self.anneal_timescale)
        return name
        
           
# ============================================================================
# BayesianQAgent class
# ============================================================================


class BayesianQAgent(TabularAgent):
    
    def __init__(self, params):
        
        # Bayesian Q-Learning agent parameters
        self.gamma = params['gamma']
        self.mu0 = params['mu0']
        self.lamda = params['lamda']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.sa_list = params['sa_list']
        self.num_mixture_samples = params['num_mixture_samples']

        # List for storing Q posterior hyperparameters
        self.Qpost_log = []

        super(BayesianQAgent, self).__init__(params['gamma'])

        # Dict for holding posterior phyperparameters
        self.Qpost = {}
        
        # Set normal-gamma prior parameters for each state-action
        for s, a in self.sa_list:

            if s not in self.Qpost: self.Qpost[s] = {}
                
            self.Qpost[s][a] = (self.mu0, self.lamda, self.alpha, self.beta)


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
            qs.append(hyp[0] + st * (hyp[3] / (hyp[1] * hyp[2]))**0.5)
            acts.append(a)
            
        return np.array(qs), np.array(acts)


    def kl_matched_hyps(self, s, a, r, s_):

        num_samples = self.num_mixture_samples

        # Find the action from s_ with the largest mean
        a_ = self.max_mu0_action(s_)

        # Parameters for next state-action NG and posterior predictive
        mu0_, lamda_, alpha_, beta_ = self.Qpost[s_][a_]
        coeff = (beta_ * (lamda_ + 1) / (alpha_ * lamda_))**0.5

        # Sample from student-t, rescale and add mean
        st = np.random.standard_t(2 * alpha_, size=(num_samples,))
        z_samp = mu0_ + st * coeff

        # Dicount and add reward
        z_samp = r + self.gamma * z_samp
    
        # z_sa posterior hyperparameters
        mu0_sa, lamda_sa, alpha_sa, beta_sa = self.Qpost[s][a]
        
        # z_sa posterior hyperparameters updated for each sample
        mu0_ = (lamda_sa * mu0_sa + z_samp) / (lamda_sa + 1)
        lamda_ = np.array([lamda_sa + 1] * mu0_.shape[0])
        alpha_ = np.array([alpha_sa + 0.5] * mu0_.shape[0])
        beta_ = beta_sa + lamda_sa * (z_samp - mu0_sa)**2 / (2 * lamda_sa + 2)

        # Sample mu and tau for each set of updated hyperparameters
        mus, taus = normal_gamma(mu0_, lamda_, alpha_, beta_)

        # MC estimates of moments
        E_tau = np.mean(taus)
        E_mu_tau = np.mean(mus * taus)
        E_mu2_tau = np.mean(mus**2 * taus)
        E_log_tau = np.mean(np.log(taus))

        # f^-1(x) where f(x) = log(x) - digamma(x)
        f_inv_term = bql_f_inv(np.log(E_tau) - E_log_tau)

        # Calculate hyperparameters of KL-matched normal gamma
        mu0 = E_mu_tau / E_tau
        lamda = 1 / (1e-12 + E_mu2_tau - E_tau * mu0**2)
        alpha = max(1 + 1e-6, f_inv_term)
        beta = alpha / E_tau

        return mu0, lamda, alpha, beta


    def max_mu0_action(self, s):
        
        # Get actions and corresponding hyperparameters of R_sa distribution
        a_mu0 = [(a, hyp[0]) for (a, hyp) in self.Qpost[s].items()]
        a, mu0 = [np.array(arr) for arr in zip(*a_mu0)]

        return a[np.argmax(mu0)]
    

    def observe(self, transition):
        t, s, a, r, s_ = transition
        self.add_observations(s, a, r, s_)
        self.last_transition = transition
        
    
    def update_after_step(self, max_buffer_length, log):

        # Log Q posterior hyperparameters
        if log: self.Qpost_log.append(deepcopy(self.Qpost))

        # Update hyperparameters
        t, s, a, r, s_ = self.last_transition
        hyps = self.kl_matched_hyps(s, a, r, s_)
        self.Qpost[s][a] = hyps
        self.last_transition = None
        
               
    def get_name(self):
        
        name = 'BayesianQAgent_gamma-{}_mu0-{}_lamda-{}_alpha-{}_beta-{}'
        
        name = name.format(self.gamma,
                           self.mu0,
                           self.lamda,
                           self.alpha,
                           self.beta)
        return name
    
        
# ============================================================================
# PSRLAgent agent definition
# ============================================================================


class PSRLAgent(TabularAgent):

    def __init__(self, params):
        
        # PSRL agent parameters
        self.gamma = params['gamma']
        self.kappa = params['kappa']
        self.mu0 = params['mu0']
        self.lamda = params['lamda']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.sa_list = params['sa_list']
        self.max_iter = params['max_iter']

        self.Ppost = {}
        self.Rpost = {}
        self.buffer = []
        self.num_s = len(set([s for (s, a) in self.sa_list]))
        self.num_a = len(set([a for (s, a) in self.sa_list]))

        # Lists for storing P and R posteriors
        self.Ppost_log = []
        self.Rpost_log = []

        super(PSRLAgent, self).__init__(params['gamma'])

        # For all sa pairs, initialise posterior hyps to prior
        for s in range(self.num_s):
            for a in range(self.num_a):
                self.Ppost[(s, a)] = self.kappa * np.ones((self.num_s,))
                
        # For valid sa pairs, initialise posterior hyps to prior
        for s, a in self.sa_list:
            for s_ in range(self.num_s):
                self.Rpost[(s, a, s_)] = (self.mu0,
                                          self.lamda,
                                          self.alpha,
                                          self.beta)
                
        self.sample_posterior_and_update_continuing_policy()


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

        for (s, a, r, s_) in self.buffer:
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
        self.buffer = []


    def take_action(self, s, t):
        return self.pi[s]


    def update_buffer(self, s, a, r, s_):
        self.buffer.append([s, a, r, s_])
        
        
    def observe(self, transition):
        t, s, a, r, s_ = transition
        self.add_observations(s, a, r, s_)
        self.buffer.append([s, a, r, s_])
    
    
    def update_after_step(self, max_buffer_length, log):
        # Log posterior values
        if log:
            self.Ppost_log.append(deepcopy(self.Ppost))
            self.Rpost_log.append(deepcopy(self.Rpost))

        if len(self.buffer) >= max_buffer_length:
            self.update_posterior()
            self.sample_posterior_and_update_continuing_policy()
           
        
    def sample_posterior_and_update_continuing_policy(self):
    
        # Sample dynamics and rewards posterior
        P, R = self.sample_posterior()
        
        # Solve Bellman equation by policy iteration
        pi, Q = solve_tabular_continuing_PI(P, R, self.gamma, self.max_iter)
        
        self.pi = pi
        
        
    def get_name(self):
        return 'PSRLAgent_gamma-{}'.format(self.gamma)
    
    

# ============================================================================
# UbeNoUnrollAgent class
# ============================================================================

            
class UbeNoUnrollAgent(TabularAgent):

    def __init__(self, params):
        
        self.Rmax = params['Rmax']
        self.kappa = params['kappa']
        self.mu0 = params['mu0']
        self.lamda = params['lamda']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.zeta = params['zeta']
        self.sa_list = params['sa_list']
        self.max_iter = params['max_iter']
        self.num_dyn_samples = params['num_dyn_samples']
        
        self.num_s = len(set([s for (s, a) in self.sa_list])) 
        self.num_a = len(set([a for (s, a) in self.sa_list]))

        super(UbeNoUnrollAgent, self).__init__(params['gamma'])

        # Set episode buffer
        self.buffer = []

        # Dynamics posterior
        self.Ppost = self.kappa * np.ones((self.num_s, self.num_a, self.num_s))

        # Rewards posterior parameters for non-allowed actions
        Rparam_ = [[[[-1e12, 1e9, 1e12, 1e9]] * self.num_s] * self.num_a] * self.num_s
        self.Rpost = np.array(Rparam_)

        Rparam = np.array([[self.mu0, self.lamda, self.alpha, self.beta]] * self.num_s)
        for (s, a) in self.sa_list:
            self.Rpost[s, a, ...] = Rparam

        self.set_Q_posterior()
        
        self.Qmu_log = []
        self.Qvar_log = []
        self.pi_log = []


    def update_buffer(self, frame):
        self.buffer.append(frame)


    def update_posterior(self):

        # Transition counts and reward sums
        p_counts = np.zeros((self.num_s, self.num_a, self.num_s))
        r_sums = np.zeros((self.num_s, self.num_a, self.num_s))
        r_counts = np.zeros((self.num_s, self.num_a, self.num_s))

        for (s, a, r, s_) in self.buffer:
            p_counts[s, a, s_] += 1
            r_sums[s, a, s_] += r
            r_counts[s, a, s_] += 1

        # Update dynamics posterior
        for s in range(self.num_s):
            for a in range(self.num_a):
                # Dirichlet posterior params are prior params plus counts
                self.Ppost[s, a] = self.Ppost[s, a] + p_counts[s, a]

        # Update rewards posterior
        for s in range(self.num_s):
            for a in range(self.num_a):
                for s_ in range(self.num_a):
                    
                    mu0, lamda, alpha, beta = self.Rpost[s, a, s_]

                    # Calculate moments
                    M1 = r_sums[s, a, s_] / max(1, r_counts[s, a, s_])
                    M2 = r_sums[s, a, s_]**2 / max(1, r_counts[s, a, s_])
                    n = r_counts[s, a, s_]

                    mu0_ = (lamda * mu0 + n * M1) / (lamda + n)
                    lamda_ = lamda + n
                    alpha_ = alpha + 0.5 * n
                    beta_ = beta + 0.5 * n * (M2 - M1**2) \
                                 + n * lamda * (M1 - mu0)**2 / (2 * (lamda + n))

                    # Update dictionary
                    self.Rpost[(s, a, s_)] = (mu0_, lamda_, alpha_, beta_)

        # Reset episode buffer
        self.buffer = []


    def set_Q_posterior(self):
        '''
            Computes the approximation (diagonal gaussian) of the Q posterior
            under policy pi.
        '''
        
        # Get expectations of P and R under posterior
        P, R = self.get_expected_P_and_R()
        
        # Compute the greedy policy and corresponding Q values
        pi, Qmu = solve_tabular_continuing_PI(P, R, self.gamma, self.max_iter)

        # Compute the uncertainty (variance) of Q
        Qvar = self.solve_bellman(self.local_rew_var,
                                  self.gamma**2,
                                  pi)

        # Set Q mean and epistemic unc. upper bound
        self.Qmu = Qmu
        self.Qvar = Qvar

        # Set policy
        self.pi = pi
        
        
    def get_expected_P_and_R(self):
        return self.Ppost / self.Ppost.sum(axis=-1)[..., None], self.Rpost[..., 0]

    
    def take_action(self, s, t, reduce_max=True):

        # Posterior mean and variance
        mu = self.Qmu[s, :]
        var = self.Qvar[s, :]

        # Sample Q from diagonal gaussian
        Q_sample = np.random.normal(loc=mu, scale=(self.zeta * var**0.5))
        
        # Return argmax to choose action
        if reduce_max:
            return np.argmax(Q_sample)
        # Return Q_sample for plotting
        else:
            return Q_sample, None


    def solve_bellman(self, local, discount, pi):
        """
            Solves BE with a custom local contribution term
        """

        s_idx = np.arange(self.num_s)
        ones = np.eye(self.num_s)
        
        # Sample dynamics (n, s, a, s_)
        P = self.sample_dynamics()        

        # Get local contribution term
        loc = local(P)
        
        # Solve linear equation
        P_ = P.mean(axis=0)[s_idx, pi]
        
        # Solution for V-like terms
        v_terms = np.linalg.solve(ones - discount * P_, loc[s_idx, pi])

        return loc + discount * np.einsum('ijk, k -> ij', P.mean(axis=0), v_terms)


    def local_rew_mean(self, P):

        means = [[[self.Rpost[(s, a, s_)][0] for s_ in range(self.num_s)]\
                                             for a in range(self.num_a)] \
                                             for s in range(self.num_s)]
        means = np.array(means)

        return np.einsum('nijk, ijk -> nij', P, means).mean(axis=0)


    def local_rew_var(self, P, each_term=False):

        # Posterior predictive of rewards (gaussian and NG) is student-t
        mu0s = [[[self.Rpost[(s, a, s_)][0] for s_ in range(self.num_s)]\
                                            for a in range(self.num_a)] \
                                            for s in range(self.num_s)]
        
        lamdas = [[[self.Rpost[(s, a, s_)][1] for s_ in range(self.num_s)]\
                                              for a in range(self.num_a)] \
                                              for s in range(self.num_s)]

        alphas = [[[self.Rpost[(s, a, s_)][2] for s_ in range(self.num_s)]\
                                              for a in range(self.num_a)] \
                                              for s in range(self.num_s)]

        betas = [[[self.Rpost[(s, a, s_)][3] for s_ in range(self.num_s)]\
                                             for a in range(self.num_a)] \
                                             for s in range(self.num_s)]

        mu0s, lamdas, alphas, betas = [np.array(a) for a in (mu0s, lamdas, alphas, betas)]

        # Epistemic uncertainty due to rewards
        mu_var = betas / (lamdas * (alphas - 1)) 
        
        # Epistemic uncertainty due to rewards and due to dynamics
        mean_mu_var = np.einsum('ijk, ijk -> ij', P.mean(axis=0), mu_var)
        var_mu_mean = np.einsum('nijk, ijk -> nij', P, mu0s).var(axis=0)

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
        self.buffer.append([s, a, r, s_])
    
    
    def update_after_step(self, max_buffer_length, log):
        
        if log:
            self.Qmu_log.append(self.Qmu[:])
            self.Qvar_log.append(self.Qvar[:])
            self.pi_log.append(self.pi[:])
            
        if len(self.buffer) >= max_buffer_length:
            self.update_posterior()
            self.set_Q_posterior()
            
    
    def update_after_episode(self):
        if len(self.buffer) > 0:
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



# ============================================================================
# MomentMatching agent definition
# ============================================================================


class MomentMatchingAgent(TabularAgent):

    def __init__(self, params):
        
        self.kappa = params['kappa']
        self.mu0 = params['mu0']
        self.lamda = params['lamda']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.zeta = params['zeta']
        self.sa_list = params['sa_list']
        self.num_PI_steps = params['num_PI_steps']
        self.num_dyn_samples = params['num_dyn_samples']
        
        super(MomentMatchingAgent, self).__init__(params['gamma'])

        # Initialise dictionaries of posterior hyperparameters
        self.Ppost = {}
        self.Rpost = {}

        # Count the number of states and actions
        self.num_s = len(set([s for (s, a) in self.sa_list])) 
        self.num_a = len(set([a for (s, a) in self.sa_list]))
        
        # Buffer for storing current episode
        self.buffer = []

        # Dynamics posterior
        self.Ppost = self.kappa * np.ones((self.num_s, self.num_a, self.num_s))

        # Rewards posterior parameters for non-allowed actions
        Rparam_ = [[[[-1e12, 1e9, 1e12, 1e9]] * self.num_s] * self.num_a] * self.num_s
        self.Rpost = np.array(Rparam_)

        # Rewards posterior parameters for allowed actions
        Rparam = np.array([[self.mu0, self.lamda, self.alpha, self.beta]] * self.num_s)
        for (s, a) in self.sa_list:
            self.Rpost[s, a, ...] = Rparam
                        

    def update_posterior(self):
        """
            Updates posterior dynamics and rewards.
        """

        # Transition counts and reward sums
        p_counts = np.zeros((self.num_s, self.num_a, self.num_s))
        r_sums = np.zeros((self.num_s, self.num_a, self.num_s))
        r_counts = np.zeros((self.num_s, self.num_a, self.num_s))

        for (s, a, r, s_) in self.buffer:
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
        self.buffer = []


    def take_action(self, s, t, reduce_max=True):
        
        # Thompson sample mean and noise scale
        mean = self.mu_z_sa[s, :]
        std = (self.val_var_val[s, :] / self.gamma**2) ** 0.5
        
        # Sample Q-values
        q_samples = np.random.normal(loc=mean, scale=self.zeta*std)
        
        if reduce_max:
            return np.argmax(q_samples)
        else:
            return q_samples, None

        
    def greedy_policy_means_and_uncertainties(self):
        """
            Returns greedy policy and corresponding means and epistemic
            uncertainties, using PI.
        """
        
        P, R = self.get_expected_P_and_R()
        
        # Get greedy policy and mu_z_sa values by PI
        pi, mu_z_sa = self.solve_tabular_continuing_PI(P, R, max_iter=self.num_pi_iter)
        
        # Solve for variances
        var_z = self.var_z(self.sample_dynamics(), pi, mu_z, R)

        # Get backed-up mean
        mu_z_sa = self.backup_mu_z(Psamp, mu_z, rew_params)

        # Get backed-up variance
        var_z_sa, var_z_sa_terms = self.backup_var_z(P, mu_z, var_z, rew_params)
            
        return pi, mu_z_sa, var_z_sa_terms

    
    def get_expected_P_and_R(self):
        return self.Ppost / self.Ppost.sum(axis=-1)[..., None], self.Rpost[..., 0]
        

#     def mu_z(self, P, pi):
#         '''
#             Return mean of z posterior    
#         '''

#         idx = np.arange(pi.shape[0])

#         # Unpack reward parameters
#         mu0, _, _, _ = rew_params 

#         # Get mean reward (num_s,)
#         P_mu0 = np.mean(np.einsum('nijk, ijk -> nij', P, mu0), axis=0)[idx, pi]
        
#         # Get mean transition matrix (num_s, num_s)
#         P_ = np.mean(P[:, idx, pi, :], axis=0)

#         # Solve for mean values of z
#         A = np.eye(P_.shape[0]) - self.gamma * P_ 
#         mu_z = np.linalg.solve(A, P_mu0)

#         return mu_z


    def backup_mu_z(self, P, mu_z):
        '''
            Return mean of z posterior by Bellman backup
        '''

        # Unpack reward parameters
        mu0, lamda, alpha, beta = (Rpost[..., i] for i in self.Rpost.shape[-1])

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

    
    def get_expected_P_and_R(self):
        return self.Ppost / self.Ppost.sum(axis=-1)[..., None], self.Rpost[..., 0]

    
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


    def update_buffer(self, s, a, r, s_):
        self.buffer.append([s, a, r, s_])
        
        
    def observe(self, transition):
        t, s, a, r, s_ = transition
        self.add_observations(s, a, r, s_)
        self.buffer.append([s, a, r, s_])
    
    
    def update_after_step(self, max_buffer_length):
        
        if len(self.buffer) >= max_buffer_length:
            
            self.update_posterior()
            if self.thompson:
                pi, mu_z_sa, var_z_sa_terms = self.greedy_policy_iteration_with_variance_output()
                self.pi, self.mu_z_sa, self.var_z_sa_terms = pi, mu_z_sa, var_z_sa_terms
            else:
                self.pi = self.ucb_policy_iteration()
    
    
    def update_after_episode(self):
        if len(self.buffer) > 0:
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
        
        name = 'MomentMatchingAgent_gamma-{}_kappa-{}_mu0-{}_lamda-{}_alpha-{}_beta-{}_zeta-{}'
        name = name.format(self.gamma,
                           self.kappa,
                           self.mu0,
                           self.lamda,
                           self.alpha,
                           self.beta,
                           self.zeta)
        
        return name
    
    


