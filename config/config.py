# External libraries
import pathlib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble, naive_bayes, tree
import numpy as np

# DIRECTORY PATHS
"""
This section defines the directory structure for consistent 
use in modules across the package.
"""
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = f"{PACKAGE_ROOT}/data"
OUTPUT_DIR = f"{PACKAGE_ROOT}/output"
DATA_FILE = 'train.csv'
PERFORMANCE_OUTPUT_FILE = 'model_performances'


# ENVIRONMENT VARIABLES
"""
This section specifies the environment variables (if any)
for use in data access, data publishing or package deployment.
If made public, make sure to NOT include any user credentials here!!
"""
ACCESS_KEY_ID = 'YOUR ACCESS KEY ID'
SECRET_ACCESS_KEY = 'YOUR SECRET ACCESS KEY'


# VARIABLE SPECIFICATIONS
"""
This section allows you to specify and group variables to simplify
referencing in the cleaning pipeline and provide an easy overview
for reviewers and auditors.
"""

CATEGORICAL_VARIABLES = []
CONTINOUS_VARIABLES = []

TARGET = 'Target'

IDENTIFIERS = ['Id', 'idhogar']

INDIVIDUAL_LEVEL = ['v18q', 
                    'dis',
                    'male', 
                    'female',
                    'estadocivil1', 
                    'estadocivil2', 
                    'estadocivil3', 
                    'estadocivil4', 
                    'estadocivil5', 
                    'estadocivil6', 
                    'estadocivil7', 
                    'parentesco1', 
                    'parentesco2',  
                    'parentesco3', 
                    'parentesco4', 
                    'parentesco5', 
                    'parentesco6', 
                    'parentesco7', 
                    'parentesco8',  
                    'parentesco9', 
                    'parentesco10', 
                    'parentesco11', 
                    'parentesco12', 
                    'instlevel1', 
                    'instlevel2', 
                    'instlevel3', 
                    'instlevel4', 
                    'instlevel5', 
                    'instlevel6', 
                    'instlevel7', 
                    'instlevel8', 
                    'instlevel9', 
                    'mobilephone', 
                    'rez_esc', 
                    'escolari', 
                    'age']


HOUSEHOLD_LEVEL = ['hacdor', 
                   'hacapo', 
                   'v14a', 
                   'refrig', 
                   'paredblolad',
                   'paredzocalo', 
                   'paredpreb',
                   'pisocemento', 
                   'pareddes', 
                   'paredmad',
                   'paredzinc', 
                   'paredfibras', 
                   'paredother', 
                   'pisomoscer', 
                   'pisoother', 
                   'pisonatur', 
                   'pisonotiene', 
                   'pisomadera',
                   'techozinc', 
                   'techoentrepiso', 
                   'techocane', 
                   'techootro', 
                   'cielorazo', 
                   'abastaguadentro', 
                   'abastaguafuera', 
                   'abastaguano',
                   'public', 
                   'planpri',
                   'noelec', 
                   'coopele', 
                   'sanitario1', 
                   'sanitario2', 
                   'sanitario3', 
                   'sanitario5',  
                   'sanitario6',
                   'energcocinar1', 
                   'energcocinar2', 
                   'energcocinar3', 
                   'energcocinar4', 
                   'elimbasu1', 
                   'elimbasu2',
                   'elimbasu3', 
                   'elimbasu4',  
                   'elimbasu6', 
                   'tipovivi1', 
                   'tipovivi2', 
                   'tipovivi3', 
                   'tipovivi4', 
                   'tipovivi5', 
                   'computer', 
                   'television', 
                   'lugar1', 
                   'lugar2',
                   'lugar3',
                   'lugar4', 
                   'lugar5', 
                   'lugar6', 
                   'area1', 
                   'area2',
                   'rooms', 
                   'r4h1',
                   'r4h2', 
                   'r4h3',
                   'r4m1',
                   'r4m2',
                   'r4m3',
                   'r4t1',  
                   'r4t2', 
                   'r4t3', 
                   'v18q1',
                   'tamhog',
                   'tamviv',
                   'hhsize',
                   'hogar_nin',
                   'hogar_adul',
                   'hogar_mayor',
                   'hogar_total',  
                   'bedrooms', 
                   'qmobilephone',
                   'v2a1', 
                   'dependency', 
                   'edjefe',
                   'edjefa', 
                   'overcrowding']


## MODELLING SPECIFICATIONS

RANDOM_STATE = 42

MODELS = {
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'param_grid': {
            'n_neighbors': list(range(1,10)),
            'leaf_size': list(range(1,10)),
            'weights': ['uniform', 'distance'],
            'p': [1,2],
            'n_jobs': [-1],
        }  
    },
    'LogisticRegression': {
        'model': LogisticRegression(),
        'param_grid': {
            'penalty': ['l2'],
            'C': list(10.**np.arange(-4., 6.)),
            'class_weight': [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
            'solver': ['lbfgs'],
            'max_iter': [1000],
            'random_state': [RANDOM_STATE]
        }
    },
    'BernoulliNB': {
        'model': naive_bayes.BernoulliNB(),
        'param_grid': {}
    },
    'GaussianNB': {
        'model': naive_bayes.GaussianNB(),
        'param_grid': {}
    },
    'DecisionTreeClassifier': {
        'model': tree.DecisionTreeClassifier(),
        'param_grid': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [2,4,6,8,10, None],
            'random_state': [RANDOM_STATE]
        }
    },
    'AdaBoostClassifier': {
        'model': ensemble.AdaBoostClassifier(),
        'param_grid': {
            'base_estimator': [
                None,
                LogisticRegression(),
                tree.DecisionTreeClassifier(),
                naive_bayes.GaussianNB()]
        }
    },
    'RandomForestClassifier': {
        'model': ensemble.RandomForestClassifier(),
        'param_grid': {
            'n_estimators': [15,25,30,35],
            'criterion': ['gini', 'entropy'],
            'max_depth': [2,4,6,None],
            'min_samples_split': [2,5,7,10,12],
            'max_features': [2,3, 'auto'],
            'random_state': [RANDOM_STATE]
        }
    }
    
}