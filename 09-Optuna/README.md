# Week 10 — Hyperparameter Optimization

## Contents
- Search strategies: grid search, random search, Bayesian optimization
- Optuna framework: `Study`, `Trial`, `suggest_*` API
- Pruning with `MedianPruner` and `SuccessiveHalvingPruner`
- Visualization: optimization history, parameter importance

## Lab
**Optuna study on MLP** — Use Optuna to automatically tune learning rate, hidden layer size, and dropout rate for an MLP on MNIST.

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    hidden = trial.suggest_int("hidden", 64, 512)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    # ... build and train model, return validation accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

---
[← Back to Course](../README.md)
