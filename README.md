# FedHybrid

This repo consists Python code for the paper "FedHybrid: A Hybrid Federated Optimization Method for Heterogeneous Clients".

## How to run the code (Examples)
You can choose 'method' from {'FedHybrid', 'FedAvg', 'GIANT', 'DiSCO', 'DiSCO_prac', 'PN_DG'}, 'dataset' from {'Quadratic_Synthetic', 'Quadratic_Housing', 'Logistic_Synthetic', 'Logistic_Mushroom'}, and 'function' from {'Quadratic', 'Logistic'}, and set parameters such as number of clients, number of dimension, stepsizes, number of clients doing Newton-type updates when running the code.

Here is an example of running FedHybird. We remark that the stepsizes used in the code will be in the power of 2; that is, when the input is a, the stepsize used in the algorithm will be 2^a.
```
python main.py --method FedHybrid --alpha1 0 --beta1 0 --alpha2 0 --beta2 0 --mu 0 --nsecond 8 --dataset Quadratic_Synthetic
```
