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
