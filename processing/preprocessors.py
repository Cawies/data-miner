# External libraries
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


"""
In this module, you define all the data cleaning transformers needed
to prepare the data for dimensionality reduction. These transformers
are the practical execution of the data cleaning requirements, located
within the assets folder. To conform to the Sklearn pipeline structure,
each transformer must be defined as an extension of the Sklearn
BaseEstimator and TransformerMixin classes.

"""

class OneHeadHouseholds(BaseEstimator, TransformerMixin):
    """ Each household may have ONLY ONE head. """

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        
        # Identify households with more or less than one head of household

        sum_head_of_household = X.groupby('idhogar')['parentesco1'].sum().reset_index()
        sum_head_of_household = sum_head_of_household[sum_head_of_household['parentesco1']!=1]

        print('{} Households have more or less than one head'.format(len(sum_head_of_household)))

        # Remove households where sum hoh != 1
        X = X[~X['idhogar'].isin(sum_head_of_household['idhogar'])].copy()

        return X


class HouseholdTargetAligner(BaseEstimator, TransformerMixin):
    """ 
    The poverty level of all individual household members must
    correspond to the poverty level reported for the head of the household.
    
    """

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()

        all_equal = X.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
        not_equal = all_equal[all_equal != True]
        print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))

        households_leader = X.groupby('idhogar')['parentesco1'].sum()

        households_no_head = X.loc[X['idhogar'].isin(households_leader[households_leader == 0].index), :]
        print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))

        households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
        print('{} Households with no head have different labels.'.format(sum(households_no_head_equal == False)))


        for household in not_equal.index:
        	true_target = int(X[(X['idhogar']==household) & (X['parentesco1']==1)]['Target'])
        	X.loc[X['idhogar'] == household, 'Target'] == true_target

        return X

class BinaryToOrdinal(BaseEstimator, TransformerMixin):
    """
    The dataset contains related binary variables, that must
    be recoded into ordinal variables.
    """

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        
        # Following variables are ordinally related and can be combined
        # to reduce feature space and high correlations among predictors

        # epared1	=1 if walls are bad
        # epared2	=1 if walls are regular
        # epared3	=1 if walls are good
        # etecho1	=1 if roof are bad
        # etecho2	=1 if roof are regular
        # etecho3	=1 if roof are good
        # eviv1	=1 if floor are bad
        # eviv2	=1 if floor are regular
        # eviv3	=1 if floor are good


        # ARGMAX --- EXTREMELY USEFUL FUNCTION FOR GENERATING ORDINAL VARIABLE FROM MULTIPLE DUMMY

        X['walls'] = np.argmax(np.array(X[['epared1', 'epared2', 'epared3']]),
                                   axis = 1)

        X['roof'] = np.argmax(np.array(X[['etecho1', 'etecho2', 'etecho3']]),
                                   axis = 1)

        X['floor'] = np.argmax(np.array(X[['eviv1', 'eviv1', 'eviv1']]),
                                   axis = 1)

        DUMMY_TO_ORDINAL = ['epared1',
                            'epared2',
                            'epared3', 
                            'etecho1', 
                            'etecho2', 
                            'etecho3', 
                            'eviv1', 
                            'eviv2', 
                            'eviv3']

        X.drop(columns=DUMMY_TO_ORDINAL, axis=1, inplace=True)


        return X


class StringsToNumerical(BaseEstimator, TransformerMixin):
    """ Convert string variables to numerical representations """

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        
        # IDENTIFY OBJECT VARIABLES
        # These have mixed data types; yes/no can be converted to 1/0

        X.select_dtypes('object')
        STRING_VARIABLES = ['dependency','edjefe','edjefa']
        X[STRING_VARIABLES]


        mapping = {"yes": 1, "no": 0}

        # Apply same operation to both X and test

        # Fill in the values with the correct mapping
        X['dependency'] = X['dependency'].replace(mapping).astype(np.float64)
        X['edjefa'] = X['edjefa'].replace(mapping).astype(np.float64)
        X['edjefe'] = X['edjefe'].replace(mapping).astype(np.float64)

        

        
        return X

class ReportAdHocCorrections(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer."""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        
        # elimbasu5 has a constant value of 0 --> remove
        X.drop(columns='elimbasu5', inplace=True)
        
        
        # Where dwelling paid and done; fillna 0
        X[X['tipovivi1']==1]['v2a1'].value_counts()
        X.loc[(X['tipovivi1']==1), 'v2a1'] = 0

        # Where dwelling precarious or assigned/borrowed fillna 'missing'
        X['v2a1'].fillna(-999999, inplace=True)
        
        
        # All missing in v18q1 have indicated 0 tablets in v18q
        # Impute all v18q1 with 0
        X['v18q'].value_counts(dropna=False)
        X[X['v18q']==0]['v18q1'].isna().sum()
        X['v18q1'].fillna(0, inplace=True)
        
        # variables that may be related:
        #    r4h1 males <12 yo
        #    r4m1 females <12 yo
        #    r4t1 persons <12 yo
        #    escolari: years of schooling
        #    hogar_nin: number of children 0-19 yo
        #    age

        # If there are no school age children, there can be no missed schooling --> set rez_esc to 0 for those

        X.loc[((X['age'] > 19) | (X['age'] < 7)) & (X['rez_esc'].isnull()), 'rez_esc'] = 0
        X['rez_esc'].fillna(-999999, inplace=True)
        
        
        return X

class AggregateHouseholds(BaseEstimator, TransformerMixin):
    """
    Aggregate individual household members' information 
    to produce aggregated household characteristics.
    """

    def __init__(self,variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):

        X = X.copy()

        ind_agg = X[self.variables+['idhogar']].groupby('idhogar').agg(['min', 'max', 'sum', 'count', 'std'])
        new_col = []
        for c in ind_agg.columns.levels[0]:
            for stat in ind_agg.columns.levels[1]:
                new_col.append(f'{c}-{stat}')
                
        ind_agg.columns = new_col

        X = X[X['parentesco1']==1].merge(ind_agg, on = 'idhogar', how = 'left')
        X.drop(columns=['meaneduc', 'SQBmeaned', 'Id', 'idhogar'], axis=1, inplace=True)

        X.fillna(0, inplace=True)

        return X