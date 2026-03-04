# PA1

## Setup

#### The environment uses Python 3.12
```bash
conda env create -f env.yml
conda activate rl_pa1
```

## Q2 - Acrobot TD Learning
- `q2.ipynb` - Main notebook with all implementations and experiments
- `analysis.ipynb` - Notebook for creating graphs and analysis.

### Main Functions

#### Training
```python
agent = SARSA(env, alpha=0.1, epsilon=0.1, decay=True, n_bins=10)
agent.train(timesteps=2000000, n=200)
```

#### Testing
```python
agent.test(qtable_path="logs/s_qtable_0.1_0.1.pkl", timesteps=1000)
```

#### Hyperparameter Sweep
```python
results = hyperparameter_sweep(env, SARSA, "s", 
    alpha_values=[0.01, 0.05, 0.1, 0.3, 0.5],
    epsilon_values=[0.01, 0.05, 0.1, 0.2, 0.3])
```

### Outputs

Logs saved in:
- `logs/` - Main experiments (including hyperparameter sweep runs)
- `decay_logs/` - Epsilon decay experiments
- `seed_logs/` - Multi-seed runs
- `c/` - Epsilon decay from 1.0 to 0.1 experiments
- `bins_logs/` - Discretization experiments

Each run saves:
- Q-table: `{algo}_qtable_{alpha}_{epsilon}.pkl`
- Metrics: `{algo}_logs_{alpha}_{epsilon}.pkl`