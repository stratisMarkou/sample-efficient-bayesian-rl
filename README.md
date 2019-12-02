# Introduction
Source for the workshop paper: E. Markou and C. E. Rasmussen, <em>Bayesian methods for efficient Reinforcement Learning in tabular problems</em>, appearing in the 2019 NIPS Workshop on Biological and Artificial RL.

We compare different Bayesian methods for representing an RL agent's uncertainty about cumulative rewards, including our own approach based on moment matching across the Bellman equations.

| Method                                        | Authors             | Paper |
| :-------------------------------------------- | :------------------ | :---- |
| Bayesian Q-Learning                           | Dearden et. al.     | [link](https://www.aaai.org/Papers/AAAI/1998/AAAI98-108.pdf)  |
| Uncertainty Bellman Equation                  | O'Donoghue et. al.  | [link](https://arxiv.org/abs/1709.05380)  |
| Posterior Sampling for Reinforcement Learning | Osband et. al.      | [link](http://papers.nips.cc/paper/5185-more-efficient-reinforcement-learning-via-posterior-sampling)  |
| Moment Matching                               | Ours                | [link](https://github.com/stratisMarkou/sample-efficient-bayesian-rl/blob/master/writeup/bayesian-methods-for-rl.pdf)  |

# Evnironments

## DeepSea

Our DeepSea MDP is a variant of the ones used in [Osband et al.](https://arxiv.org/abs/1608.02731). The agent starts from the left-most state and can choose swim-*left* or swim-*right* from each of the N states in the environment. Swim-*left* always succeeds and moves the agent to the left, giving r = 0 (red transitions). Swim-*right* from s = 1, ..., (N - 1) succeeds with probability (1 - 1/N), moving the agent to the right and otherwise fails moving the agent to the left (blue arrows), giving r ~ N(-δ, δ^2) regardless of whether it succeeds. A successful swim-*right* from s = N moves the agent back to s = 1 and gives r = 1. We choose δ so that *right* is optimal for up to N = 40.

<p align="center">
  <img src="writeup/png/environments-deepsea.png" align="middle" width="800" />
</p>

This environment is designed to test whether the agent continues exploring despite receiving negative rewards. Sustained exploration becomes increasingly important for large N. As argued in (Osband's thesis)[https://searchworks.stanford.edu/view/11891201], in order to avoid exponentially poor performance, exploration in such chain-like environments must be guided by uncertainty rather than randomness.



## WideNarrow

The WideNarrow MDP has 2N + 1 states and deterministic transitions. Odd states except s = (2N + 1) have W actions, out of which one gives r ~ N(μ1, σ1^2) whereas all others give r ~ N(μ2, σ2^2), with μ2 < μ1. Even states have a single action also giving r ~ N(μ2, σ^2). In our experiments we use μ1 = 0.5, μ2 = 0 and σ1 = σ2 = 1.

<p align="center">
  <img src="writeup/png/environments-widenarrow.png" align="middle" width="500" />
</p>

where $\btheta$ loosely denotes all modelling parameters, $\s'$ denotes the next-state from $(\s_1, \ac_1)$, $\s{''}$ denotes the next-state from $(\s_1, \ac_2)$ and a', a'' denote the corresponding next-actions. Although the remaining three terms are non-zero under the posterior, BQL, UBE and MM ignore them, instead sampling from a factored posterior. The WideNarrow environment enforces strong correlations between these state actions, allowing us to test the impact of a factored approximation.

## PriorMDP

The aforementioned MDPs have very specific and handcrafted dynamics and rewards, so it is interesting to also compare the algorithms on environments which lack this sort of structure. For this we sample finite MDPs with Ns states and Na actions from a prior distribution, as in [Osband](http://papers.nips.cc/paper/5185-more-efficient-reinforcement-learning-via-posterior-sampling). $\mct$ is a Categorical with parameters $\{\bs{\eta_{\s, \ac}}\}$ with:
\begin{align*}
\bs{\eta}_{\s, \ac} \sim \text{Dirichlet}(\bs{\kappa}_{\s, \ac}),
\end{align*}
with pseudo-count parameters $\bs{\kappa}_{\s, \ac} = \bm{1}$, while $\mcr \sim \mc{N}(\mu_{\s, \ac}, \tau_{\s, \ac}^{-1})$ with:
\begin{align*}
\mu_{\s, \ac}, \tau_{\s, \ac} \sim NG(\mu_{\s, \ac}, \tau_{\s, \ac} | \mu, \lambda, \alpha, \beta) \text{ with } (\mu, \lambda, \alpha, \beta) = (0.00, 1.00, 4.00, 4.00).
\end{align*}

We chose these hyperparameters because they give Q*-values in a reasonable range.

# Results


## Regret summaries

<p align="center">
  <img src="writeup/png/regret_summary_deepsea.png" align="middle" width="600" />
</p>

<p align="center">
  <img src="writeup/png/regret_summary_widenarrow.png" align="middle" width="600" />
</p>

<p align="center">
  <img src="writeup/png/regret_summary_priormdp.png" align="middle" width="600" />
</p>

## Posterior evolutions on PriorMDP

<p align="center">
  <img src="writeup/png/bql-0_0-4_0-3_0-3_0-posterior-priormdp-4-2-seed-0.png" align="middle" width="400" />
  <img src="writeup/png/psrl-0_0-4_0-3_0-3_0-posterior-priormdp-4-2-seed-0.png" align="middle" width="400" />
</p>

<p align="center">
  <img src="writeup/png/ube-0_0-4_0-3_0-3_0-0_1-posterior-priormdp-4-2-seed-0.png" align="middle" width="400" />
  <img src="writeup/png/mm-0_0-4_0-3_0-3_0-1_0-posterior-priormdp-4-2-seed-0.png" align="middle" width="400" />
</p>

# Conclusions
