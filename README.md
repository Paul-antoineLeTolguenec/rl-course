# Reinforcement Learning Course

Interactive course site — [live](https://paul-antoineletolguenec.github.io/rl-course/)

**Paul-Antoine LE TOLGUENEC**

---

## Course Layout — 3 hours

### Hour 1 — Introduction to Reinforcement Learning & Bandit Algorithms
- History of RL: from early foundations to modern achievements
- Core concepts: agent, environment, state, action, reward
- Exploration vs exploitation
- Bandits: history, formalism, objective, ε-greedy, UCB, Thompson Sampling
- Bridge: contextual bandits → motivation for Markov Decision Processes
- `01_bandits.ipynb`

*5 min break*

### Hour 2 — Markov Decision Processes & Dynamic Programming
- Markov Decision Processes: formalism, Bellman equations
- Dynamic Programming: policy evaluation, policy improvement, policy iteration, value iteration
- Model-Free: Monte Carlo, TD learning, Q-learning, SARSA
- `02_markov_chains.ipynb`
- `03_dynamic_programming.ipynb`
- `03b_model_free.ipynb` — optional

*5 min break*

### Hour 3 — Policy Search & Deep Reinforcement Learning
- Policy Search: Reinforce, Cross-Entropy Method, Evolution Strategies
- Approximate Dynamic Programming & Deep Reinforcement Learning: function approximation, Deep Q-Network
- `04_policy_search.ipynb`
- `05_deep_rl.ipynb`

---

## Notebooks

| # | Notebook | Chapter | Optional |
|---|----------|---------|----------|
| 01 | `01_bandits.ipynb` | Bandits | No |
| 02 | `02_markov_chains.ipynb` | MDP | No |
| 03 | `03_dynamic_programming.ipynb` | DP | No |
| 03b | `03b_model_free.ipynb` | Model-Free | Yes |
| 04 | `04_policy_search.ipynb` | Policy Search | No |
| 05 | `05_deep_rl.ipynb` | Deep RL | No |

## Local Preview

```sh
python3 -m http.server
# or
npx serve .
```
