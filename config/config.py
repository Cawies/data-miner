import pathlib
import os

# DIRECTORY PATHS
"""
This section defines the directory structure for consistent 
use in modules across the package.
"""
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = f"{PACKAGE_ROOT}/data"
OUTPUT_DIR = f"{PACKAGE_ROOT}/output"
DATA_FILE = 'train.csv'


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