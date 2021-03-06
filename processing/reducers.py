# External libraries
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE

# BUG: One of the reducers output different set of variables for each run
# TODO: Set seed/random_state to ensure replicability

"""
This module defines the methods for reducing the dataset using
statistical and machine learning techniques. The association between
all independent variables and the target dependent variable is evaluated
for the purpose of discarding independent variables which exhibit a 
multicolinearity or a weak relationship to the dependent variable.

The following methods are included:

1. Variance evaluation.
2. Duplication
3. Correlation and multicolinearity
4. Entropy
5. Decision Tree Receiver Operating characteristic Curve (ROC)

"""

class EvaluateConstants(BaseEstimator, TransformerMixin):
    """ evaluate variance ."""

    def __init__(self, variables=None, threshold=0.01) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        print('CONSTANTS')

        X = X.copy()
        
        constants = VarianceThreshold(threshold=self.threshold)
        constants.fit(X)
        constant_variables = [var for var in X.columns 
                              if var not in X.columns[constants.get_support()]]

        print('Number of variables evaluated: ', len(X.columns))
        X = X.drop(columns=constant_variables, axis=1)




        print('Number of variables identified for removal: ', len(constant_variables))
        print('Number of variables remaining: ', len(X.columns))
        print('_________________________________________________')

        return X

class RemoveDuplicates(BaseEstimator, TransformerMixin):
    """ identify identical variables """

    def __init__(self, variables=None, threshold=0.01) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        print('DUPLICATES')

        X = X.copy()
        
        duplicated_variables = []
    
        for variable in range(0, len(X.columns)):
            var_1 = X.columns[variable]

            for var_2 in X.columns[variable + 1:]:
                if X[var_1].equals(X[var_2]):
                    #print(var_1, ' = ', var_2)
                    duplicated_variables.append(var_2)

        print('Number of variables evaluated: ', len(X.columns))
        print('Number of variables identified for removal: ', len(duplicated_variables))
        X.drop(columns=duplicated_variables, axis=1, inplace=True)
        print('Number of variables remaining: ', len(X.columns))
        print('_________________________________________________')
        
        
        
        return X

class IdentifyCorrelatedPredictors(BaseEstimator, TransformerMixin):
    """ identify groups of correlated variables."""

    def __init__(self, variables=None, threshold=0.01) -> None:
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

        print('CORRELATIONS')
        
        # Identify groups of correlated variables
        correlations = X.corr().abs().unstack().sort_values(ascending=False)
        correlations = pd.DataFrame(correlations[(correlations>=0.8) & (correlations<1)]).reset_index()
        correlations.columns = ['var_1', 'var_2', 'r']

        grouped_variable_ls = []
        correlated_groups = []

        for variable in correlations['var_1'].unique():
            if variable not in grouped_variable_ls:
                correlated_block = correlations[correlations['var_1'] == variable]
                grouped_variable_ls = grouped_variable_ls + list(correlated_block['var_2'].unique()) + [variable]

                correlated_groups.append(correlated_block)
        print(f"Number of variables evaulated: {len(X.columns)}")       
        print(f"Number of correlated groups found: {len(correlated_groups)}")

        # Select one variable from each group with highest predictive value
        hold_importances = []
        for group in correlated_groups:
            variables = list(list(group['var_2'].unique())+list(group['var_1'].unique()))

            random_forest = RandomForestClassifier(n_estimators=200, 
                                                   random_state=42, 
                                                   max_depth=4)

            random_forest.fit(X[variables], X['Target'])

            importances = pd.concat([pd.Series(variables), pd.Series(random_forest.feature_importances_)], axis=1)

            importances.columns = ['variable', 'importance']
            importances.sort_values(by='importance', ascending=False, inplace=True)
            hold_importances.append(importances.reset_index(drop=True))

        variables_to_drop = []
        for group in hold_importances:
            variables_to_drop.append(list(group['variable'].iloc[1:]))

        variables_to_drop = list(set(sum(variables_to_drop, [])))
        print(f"Number of variables identified for removal: {len(variables_to_drop)}")
        X.drop(columns=variables_to_drop, axis=1, inplace=True)

        print(f"Number of variables remaining: {len(X.columns)}")
        print('_________________________________________________')

        return X

class EvaluateEntropy(BaseEstimator, TransformerMixin):
    """ Evaluate Entropy """

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

        print('ENTROPY')

        X = X.copy()
        
        sel_ = SelectKBest(mutual_info_classif, k=50).fit(X.drop('Target', axis=1), X['Target'])
        variables_to_keep = list(X.drop('Target', axis=1).columns[sel_.get_support()])
        print(f"Number of variables evaluated: {len(X.columns)}")
        print(f"Keeping the 50 most predictive variables.")
    
        X = X[variables_to_keep+['Target']].copy()
        print(f"Number of remaining variables: {len(X.columns)}")
        print('_________________________________________________')
        
        return X

class DecisionTreeRoC(BaseEstimator, TransformerMixin):
    """ Decision Tree using ROC curve for variable importance."""

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

        print('DECISION TREE ROC')

        X = X.copy()
        
        roc_values = []


        print(f"Number of variables evaluated: {len(X.columns)}")
    
        for variable in X.drop(columns='Target').columns:
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(X[variable].to_frame(), X['Target'])
            y_scored = clf.predict_proba(X[variable].fillna(0).to_frame())
            roc_values.append(roc_auc_score(X['Target'], y_scored, multi_class='ovo'))

        roc_values = pd.Series(roc_values)
        roc_values.index = X.drop(columns='Target').columns
        variables_to_keep = list(roc_values[roc_values>roc_values.mean()].index)
        X = X[variables_to_keep+['Target']].copy()
        print(f"Number of variables remaining: {len(X.columns)}")
        
        return X

class RecursiveRandomForestImportance(BaseEstimator, TransformerMixin):
    """ 
    Evaluate importance of variable in Random Forest Classifier
    using recursive selection.
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

        print('RANDOM FOREST CLASSIFIER')

        X = X.copy()
        rf = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=12)
        rf.fit(X.drop('Target', axis=1), X['Target'])
        print(f"Number of variables evaluated: {len(X.columns)}")

        variables_to_keep = X.drop('Target', axis=1).columns[(rf.get_support())]
        X = X[list(variables_to_keep)+['Target']].copy()
        print(f"Number of variables remaining: {len(X.columns)}")
        
        return X

class ExportReducedData(BaseEstimator, TransformerMixin):
    def __init__(self,output_folder=None):
        if not isinstance(output_folder, str):
            self.output_folder = str(output_folder)
        else:
            self.output_folder = output_folder

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):

        X = X.copy()
        X.to_excel(f"{self.output_folder}/reduced_data.xlsx", index=False)
        print('Exporting reduced dataset...')
        print('End pipeline.')

        return X