
import numpy as np
import random

from sklearn.datasets import load_boston
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import json
import pandas as pd

# parameters to determine the distribution of sampling
class SampleParam: 
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

# generate synthetic data, linear regression model
class QuadraticDataGen: 
    def __init__(self, nclient, ndim, nsample_param, scaling_param, prior_param, ncond_param):
        self.nclient = nclient
        self.ndim = ndim
        self.nsample_param = nsample_param
        self.scaling_param = scaling_param
        self.prior_param = prior_param
        self.ncond_param = ncond_param # parts of the clients have larger condition numbers

    def run(self):
        print('local datasizes')
        ni = np.random.lognormal(self.nsample_param.mean, self.nsample_param.sigma, self.nclient).astype(int) + 50 # number of samples per client
        print(ni)

        A = [[] for _ in range(self.nclient)]
        y = [[] for _ in range(self.nclient)]

        #### define some eprior ####
        x_0 = np.random.normal(self.prior_param.mean, self.prior_param.sigma, self.ndim)
        scaling = np.random.lognormal(self.scaling_param.mean, self.scaling_param.sigma, self.nclient) # how diverse the data is

        client_cond = random.sample(range(self.nclient), self.ncond_param) # parts of the clients have larger condition numbers
        client_normal = np.setdiff1d(range(self.nclient), client_cond) 
        print('client_cond', client_cond)

        print('condition number of local Hessians')
        for i in client_normal:
            Ai = np.random.rand(ni[i], self.ndim) * scaling[i]
            print(np.linalg.eigh(np.transpose(Ai).dot(Ai))[0][-1])
            
            sig = np.random.random(1)
            v = np.random.normal(0, sig, ni[i]) 
            yi = Ai.dot(x_0) + v
            yi = yi.reshape([ni[i], 1])
            A[i] = Ai
            y[i] = yi
        
        for i in client_cond:
            ai = np.random.rand(self.ndim, 1)
            ai[:int(self.ndim/2)] = [ai[i]*1e2 for i in range(int(self.ndim/2))]
            ai[int(self.ndim/2):] = [ai[i+int(self.ndim/2)]*1e-2 for i in range(self.ndim-int(self.ndim/2))]
            Ai = np.random.rand(ni[i], self.ndim).dot(np.diag(ai.T[0])) * scaling[i] 
            print(np.linalg.eigh(np.transpose(Ai).dot(Ai))[0][-1])
            
            sig = np.random.random(1)
            v = np.random.normal(0, sig, ni[i]) 
            yi = Ai.dot(x_0) + v
            yi = yi.reshape([ni[i], 1])
            A[i] = Ai
            y[i] = yi

        return A, y

# real data, boston housing, linear regression model
class QuadraticHousing: 
    def __init__(self, nclient):
        self.nclient = nclient

    def run(self):
        boston_dataset = load_boston()
        boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
        boston['MEDV'] = boston_dataset.target

        X = boston.drop('MEDV', axis = 1)
        y = boston['MEDV']  

        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

        # Not uniformly distributed among clients

        X_train, y_train = shuffle(X_train, y_train)
        X_df = X_train.copy()
        X_df['y_train'] = y_train
        X_df = X_df.reset_index()
        X_df = X_df.drop('index', axis = 1)

        X_df_sorted = X_df.sort_values(by = ['y_train'])
        X_df_sorted = X_df_sorted.reset_index()

        X_train = X_df.drop('y_train', axis = 1)
        y_train = X_df['y_train']

        # number of samples
        n = len(y_train)
        # ndim
        ndim = X_train.shape[1] + 1 

        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
        intercept = np.ones((X_test.shape[0], 1))
        X_test = np.hstack((intercept, X_test))

        # number of samples at each client
        p = np.zeros(self.nclient)
        s = 0
        for i in range(self.nclient):
            p[i] = random.uniform(0.1, 1)
            s += p[i]
        p = p/s
        ni = p * n
        ni = [int(num) for num in ni]
        ni[-1] = ni[-1] + n - sum(ni)

        indices_set = [[] for i in range(self.nclient)]
        s = 0
        for j in range(self.nclient):
            indices_set[j] =  X_df_sorted['index'].to_list()[s:s+ni[j]]
            s+= ni[j]

        A = [[] for _ in range(self.nclient)]
        y = [[] for _ in range(self.nclient)]
        for i in range(self.nclient):
            A[i] = X_train[indices_set[i]]
            y[i] = y_train.to_numpy()[indices_set[i]]

        y_test = y_test.to_numpy()

        return A, y, X_test, y_test, ndim        


# generate synthetic data, logistic regression model
class LogisticDataGen:
    def __init__(self, nclient, ndim, nclass, nsample_param, scaling_param, prior_param):
        self.nclient = nclient
        self.ndim = ndim
        self.nclass = nclass
        self.nsample_param = nsample_param
        self.scaling_param = scaling_param
        self.prior_param = prior_param
        #self.ncond_param = ncond_param # parts of the clients have larger condition numbers

    def softmax(self, z):
        ex = np.exp(z)
        sum_ex = np.sum(np.exp(z))
        return ex/sum_ex

    def run(self):
        print('local datasizes')
        ni = np.random.lognormal(self.nsample_param.mean, self.nsample_param.sigma, self.nclient).astype(int) + 50 # number of samples per client
        print(ni)

        A = [[] for _ in range(self.nclient)]
        y = [[] for _ in range(self.nclient)]

        #### define some eprior ####
        mean_W = np.random.normal(self.prior_param.mean, self.prior_param.sigma, self.nclient)
        mean_b = mean_W
        mean_x = np.random.normal(self.prior_param.mean, self.prior_param.sigma, self.nclient)

        mean_x = np.zeros((self.nclient, self.ndim))
        for i in range(self.nclient):
            mean_x[i] = np.random.normal(mean_x[i], 1, self.ndim)

        diagonal = np.zeros(self.ndim)
        for j in range(self.ndim):
            diagonal[j] = np.power((j+1), -1.2)
        cov_x = np.diag(diagonal)

        for i in range(self.nclient):

            W = np.random.normal(mean_W[i], 1, (self.ndim, self.nclass))
            b = np.random.normal(mean_b[i], 1,  self.nclass)

            Ai = np.random.multivariate_normal(mean_x[i], cov_x, ni[i])
            yi = np.zeros(ni[i])

            for j in range(ni[i]):
                tmp = np.dot(Ai[j], W) + b
                yi[j] = np.argmax(self.softmax(tmp))

            A[i] = Ai.tolist()
            y[i] = yi.tolist()

        return A, y

    def split_data(self):

        train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
        test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

        A, y = self.run() # synthetic (0.5, 0.5)

        for i in range(self.nclient):

            uname = 'f_{0:05d}'.format(i)        
            combined = list(zip(A[i], y[i]))
            random.shuffle(combined)
            A[i][:], y[i][:] = zip(*combined)
            num_samples = len(A[i])
            train_len = int(0.9 * num_samples)
            test_len = num_samples - train_len
    
            train_data['users'].append(uname) 
            train_data['user_data'][uname] = {'x': A[i][:train_len], 'y': y[i][:train_len]}
            train_data['num_samples'].append(train_len)
            test_data['users'].append(uname)
            test_data['user_data'][uname] = {'x': A[i][train_len:], 'y': y[i][train_len:]}
            test_data['num_samples'].append(test_len)
    
        with open('Logistic_Synthetic/train_data05.json', 'w') as fp:
            json.dump(train_data, fp)
        with open('Logistic_Synthetic/test_data05.json', 'w') as fp:
            json.dump(test_data, fp)
        
    def saved_data(self):
        with open('Logistic_Synthetic/train_data05.json', 'r') as fp:
            train_data = json.load(fp)
        with open('Logistic_Synthetic/test_data05.json', 'r') as fp:
            test_data = json.load(fp)
        
        df_train = pd.DataFrame.from_dict(train_data['user_data'])
        df_train = df_train.T
        df_test = pd.DataFrame.from_dict(test_data['user_data'])
        df_test = df_test.T

        A_train = df_train['x'].to_numpy()
        y_train = df_train['y'].to_numpy()

        A_test = df_test['x'].to_numpy()
        y_test = df_test['y'].to_numpy()

        for i in range(self.nclient):
            A_train[i] = np.array(A_train[i])
            y_train[i] = np.array(y_train[i])

        A_test_all = np.concatenate([A_test[i] for i in range(self.nclient)])
        y_test_all = np.concatenate([y_test[i] for i in range(self.nclient)])

        return A_train, y_train, A_test_all, y_test_all


# real data, mushroom, logistic regression model
class LogisticMushroom: 
    def __init__(self, nclient):
        self.nclient = nclient

    def run(self):
        # reading data files
        df =  pd.read_table('agaricus-lepiota.data', delimiter=',', header=None)
        column_labels = [
            'class', 'cap shape', 'cap surface', 'cap color', 'bruised', 'odor',
            'gill attachment', 'gill spacing', 'gill size', 'gill color', 
            'stalk shape', 'stalk root', 'stalk surface above ring',
            'stalk surface below ring', 'stalk color above ring',
            'stalk color below ring', 'veil type', 'veil color', 'ring number',
            'ring type', 'spore print color', 'population', 'habitat']

        df.columns = column_labels
        # excluding any training example that has missing values for stalk root.
        df = df[df['stalk root'] != '?']

        X = df.loc[:, df.columns != 'class']
        y = df['class'].to_frame()

        # Encoding categorical features
        X_enc = pd.get_dummies(X)
        # Standardizing the features
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_enc)

        # Encoding the target variable
        le = LabelEncoder()
        y_enc = le.fit_transform(y.values.ravel())

        X_train, X_test, y_train, y_test = train_test_split(X_std, y_enc, test_size=0.3, stratify=y_enc, random_state=42)

        # number of examples
        n = X_train.shape[0]
        # ndim
        ndim = X_train.shape[1] + 1
        # number of classes
        k = 2
        label = [0,1]

        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
        intercept = np.ones((X_test.shape[0], 1))
        X_test = np.hstack((intercept, X_test))

        # Not uniformly distributed among clients

        X_train, y_train = shuffle(X_train, y_train)

        X_df = pd.DataFrame([y_train, X_train])
        X_df = X_df.T
        X_df.columns = ['y_train', 'X_train']
        X_df = X_df.reset_index()

        # Split it into different groups according to the label
        X_df_0 = X_df[X_df['y_train']==0]
        X_df_1 = X_df[X_df['y_train']==1]

        p = np.zeros(self.nclient)
        s = 0
        for i in range(self.nclient):
            p[i] = np.random.lognormal(4, 2) + 50
            s += p[i]
        p = p/s

        # number of samples at each client
        ni = p * n
        ni = [int(num) for num in ni]
        ni[-1] = ni[-1] + n - sum(ni) 

        X_df_reorder = pd.concat([X_df_0, X_df_1])

        indices_set = [[] for i in range(self.nclient)]
        s = 0
        for j in range(self.nclient):
            indices_set[j] =  X_df_reorder['index'].to_list()[s:s+ni[j]]
            s += ni[j]

        A = [[] for _ in range(self.nclient)]
        y = [[] for _ in range(self.nclient)]
        for i in range(self.nclient):
            A[i] = X_train[indices_set[i]]
            y[i] = y_train[indices_set[i]]

        return A, y, X_test, y_test, ndim 
