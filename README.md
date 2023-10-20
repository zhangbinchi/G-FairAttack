# G-FairAttack

The code is associated with *Adversarial Attacks on Fairness of Graph Neural Networks*.

## Requirements

- python == 3.9.13
- torch == 1.11.0
- torch-geometric == 2.0.4
- numpy == 1.21.5
- numba == 0.56.3
- networkx == 2.8.4
- scikit-learn == 1.1.1
- scipy == 1.9.1
- dgl == 0.9.1
- deeprobust == 0.2.5

## Attack
Conduct fairness evasion attack on the structure of Facebook dataset.
```
python attack.py --dataset facebook
```
Conduct fairness poisoning attack on the structure of Facebook dataset.
```
python attack.py --dataset facebook --poisoning
```
Please refer to Appendix E for the hyperparameter settings.

## Evaluation
After running the attack program, replace the 'Path' string in each test program with the real path of the poisoned adjacency matrix.

Evaluate the attack on a victim model (e.g., GCN)

1. Replace the 'Path' at line 53 in test_gcn_evasion.py with the path of the perturbed adjacency matrix.
2. Run
```
python test_gcn_evasion.py
```
Please refer to Appendix E for the hyperparameter settings.
