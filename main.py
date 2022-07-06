import numpy as np
import random
import pandas as pd
import argparse
from scipy.optimize import minimize

from Dataset import SampleParam, QuadraticDataGen, QuadraticHousing, LogisticDataGen, LogisticMushroom
from CostFunc import QuadraticCostFunc, LogisticCostFunc
from methods import TrainParam, TuneParam, FedAvg, GIANT, DiSCO, FedHybrid, PN_DG

def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dataset =='Quadratic_Synthetic':
        data_generator = QuadraticDataGen(nclient=args.nclient, 
                             ndim=args.ndim, 
                             nsample_param=SampleParam(args.mean_nsample_param, args.sigma_nsample_param), 
                             scaling_param=SampleParam(args.mean_scaling_param, args.sigma_scaling_param), 
                             prior_param=SampleParam(args.mean_prior_param, args.sigma_prior_param),
                             ncond_param = args.ncond_param
                             )
        
        A, y =  data_generator.run()

    if args.dataset =='Quadratic_Housing':
        data_generator = QuadraticHousing(nclient=args.nclient)

        A, y, _, _, ndim = data_generator.run()
        args.ndim = ndim

    if args.dataset =='Logistic_Synthetic':
        data_generator = LogisticDataGen(nclient=args.nclient, 
                             ndim=args.ndim,
                             nclass=args.nclass,
                             nsample_param=SampleParam(args.mean_nsample_param, args.sigma_nsample_param), 
                             scaling_param=SampleParam(args.mean_scaling_param, args.sigma_scaling_param), 
                             prior_param=SampleParam(args.mean_prior_param, args.sigma_prior_param),
                             )   
        data_generator.split_data()
        A, y, _, _ = data_generator.saved_data()

    if args.dataset =='Logistic_Mushroom':
        data_generator = LogisticMushroom(nclient=args.nclient)

        A, y, _, _, ndim = data_generator.run()
        args.ndim = ndim
        

    if args.function == 'Quadratic':
        func = QuadraticCostFunc(A, y, gamma=args.gamma)

    if args.function == 'Logistic':
        func = LogisticCostFunc(A, y, gamma=args.gamma)

    # initial point
    initial_x = np.random.rand(args.ndim, 1)
    print('initial_x',initial_x)
    #initial_x = np.zeros([args.ndim, 1])
    client_gradient = None
    client_Newton = None

    # calculate the optimal fn_star
    fn_min = minimize(func.global_func, initial_x, tol=1e-30)
    fn_star = fn_min.fun

    if args.method == 'FedAvg':
        method = FedAvg(func, fn_star)

    if args.method == 'GIANT':
        method = GIANT(func, fn_star)

    if args.method == 'DiSCO':
        method = DiSCO(func, fn_star)

    if args.method == 'FedHybrid':
        client_Newton = random.sample(range(args.nclient), args.nsecond) # clients that perform 2nd updates
        client_gradient = np.setdiff1d(range(args.nclient), client_Newton) # clients that perform 1st updates
        client_Newton = np.setdiff1d(range(args.nclient), client_gradient)
        print('client_gradient', client_gradient, 'client_Newton', client_Newton)
        method = FedHybrid(func, fn_star)

    if args.method == 'PN_DG':
        method = PN_DG(func, fn_star)

    if args.mode == "train":
        fn_list, k_iter, x_last = method.train(param = TrainParam(alpha1=args.alpha1,
                                                        beta1=args.beta1,
                                                        alpha2=args.alpha2,
                                                        beta2=args.beta2,
                                                        mu=args.mu,
                                                        K=args.K,
                                                        client_gradient=client_gradient,
                                                        client_Newton=client_Newton,
                                                        initial_x=initial_x
                                                        ))
        
        # save data
        df = pd.DataFrame(fn_list)
        df.to_csv(args.dataset + '/Result/' + args.method + '_' + str(args.alpha1) + '_' + str(args.beta1) + '_' 
                    + str(args.alpha2) + '_' + str(args.beta2) + '_' + str(args.mu) + '_' + str(args.nsecond) + '_' + str(args.seed) + '_' + str(client_Newton) + '.csv')
        

    if args.mode == "tune":
        tune_result = method.tune(param = TuneParam(alpha1_range=args.alpha1_range,
                                                    beta1_range=args.beta1_range,
                                                    alpha2_range=args.alpha2_range,
                                                    beta2_range=args.beta2_range,
                                                    mu_range=args.mu_range,
                                                    K=args.K,
                                                    client_gradient=client_gradient,
                                                    client_Newton=client_Newton,
                                                    initial_x=initial_x
                                                    ))
        # save data
        df = pd.DataFrame(tune_result)
        df.to_csv(args.dataset + '/tune/' + args.method + '_' + str(args.nsecond) + '_' + str(args.seed) + '_tune.csv')


if __name__ == '__main__':
    # parser start
    parser = argparse.ArgumentParser(description='PyTorch')
    
    parser.add_argument('--dataset', type=str, default='Quadratic_Housing') # 'Quadratic_Synthetic', 'Quadratic_Housing', 'Logistic_Synthetic', 'Logistic_Mushroom'
    parser.add_argument('--function', type=str, default='Quadratic') # 'Quadratic', 'Logistic'

    parser.add_argument('--nclient', type=int, default=8) # 10 for 'Quadratic_Synthetic' and 'Logistic_Synthetic', 8 for 'Quadratic_Housing' and 'Logistic_Mushroom'
    parser.add_argument('--ndim', type=int, default=12) # 3 for 'Quadratic_Synthetic', 12 for 'Logistic_Synthetic'
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--mean_nsample_param', type=float, default=4)
    parser.add_argument('--sigma_nsample_param', type=float, default=2)
    parser.add_argument('--mean_scaling_param', type=float, default=2) # 2 for quadratic 
    parser.add_argument('--sigma_scaling_param', type=float, default=4) # 4 for quadratic
    parser.add_argument('--mean_prior_param', type=float, default=0)
    parser.add_argument('--sigma_prior_param', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1) # 1e-2 for quadratic, 1 for logistic
    parser.add_argument('--ncond_param', type=float, default=4) # the number of clients that have larger condition numbers

    parser.add_argument('--method', type=str, default='FedHybrid')
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--alpha1', type=float, default=None)
    parser.add_argument('--beta1', type=float, default=None)
    parser.add_argument('--alpha2', type=float, default=None)
    parser.add_argument('--beta2', type=float, default=None)
    parser.add_argument('--mu', type=float, default=None)
    parser.add_argument('--K', type=int, default=500)
    parser.add_argument('--nsecond', type=int, default=None)

    # for tuning
    parser.add_argument('--alpha1_range', nargs='+', type=int, default=None)
    parser.add_argument('--beta1_range', nargs='+', type=int, default=None)
    parser.add_argument('--alpha2_range', nargs='+', type=int, default=None)
    parser.add_argument('--beta2_range', nargs='+', type=int, default=None)
    parser.add_argument('--mu_range', nargs='+', type=int, default=None)

    args = parser.parse_args()
    main(args)