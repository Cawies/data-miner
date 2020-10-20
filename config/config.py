import pathlib
import os

# DIRECTORY PATHS
"""
This section defines the directory structure for consistent 
use in modules across the package.
"""
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = f"{PACKAGE_ROOT}/data"
DATA_FILE = 'train.csv'


# ENVIRONMENT VARIABLES
"""
This section specifies the environment variables (if any)
for use in data access, data publishing or package deployment.
If made public, make sure to NOT include any user credentials here!!
"""
ACCESS_KEY_ID = 'Dont store in source code'
SECRET_ACCESS_KEY = 'Dont store in source code'


# VARIABLE SPECIFICATIONS
"""
This section allows you to specify and group variables to simplify
referencing in the cleaning pipeline and provide an easy overview
for reviewers and auditors.
"""
IDENTIFIERS = []
CATEGORICAL_VARIABLES = []
CONTINOUS_VARIABLES = []