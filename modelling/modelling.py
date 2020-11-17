# External libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate, ShuffleSplit

# Internal modules
from config import config

def BaseLineModels(models, X, y):
    
    row_index = 0
    cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 42)
    baseline_output = pd.DataFrame(columns=['model', 'mean_train_acc', 'mean_test_acc', 'parameters'])
    
    for model in [models[key]['model'] for key in models]:
        baseline_output.loc[row_index, 'model'] = model.__class__.__name__
        cross_validation_result = cross_validate(model, X, y, cv = cv_split, return_train_score=True, scoring='f1')
        model_parameters = model.fit(X, y).get_params()
        
        baseline_output.loc[row_index, 'mean_train_acc'] = cross_validation_result['train_score'].mean()
        baseline_output.loc[row_index, 'mean_test_acc'] = cross_validation_result['test_score'].mean()
        baseline_output.loc[row_index, 'parameters'] = [model_parameters]
        row_index+=1
        
        
        
    baseline_output.sort_values(by='mean_test_acc', ascending=False, inplace=True)
    
    row_index = 0
    tuned_output = pd.DataFrame(columns=['model', 'mean_train_acc_tuned', 'mean_test_acc_tuned', 'parameters_tuned'])
    
    for model in [models[key] for key in models]:
        tuned_output.loc[row_index, 'model'] = model['model'].__class__.__name__
        tuned_model = RandomizedSearchCV(model['model'], param_distributions=model['param_grid'], scoring = 'f1', cv = cv_split, return_train_score=True)
        tuned_model.fit(X, y)

        tuned_output.loc[row_index, 'mean_train_acc_tuned'] = tuned_model.cv_results_['mean_train_score'][tuned_model.best_index_]
        tuned_output.loc[row_index, 'mean_test_acc_tuned'] = tuned_model.cv_results_['mean_test_score'][tuned_model.best_index_]
        tuned_output.loc[row_index, 'parameters_tuned'] = [tuned_model.best_params_]
        row_index+=1

    

    output = baseline_output.join(tuned_output.set_index('model'), on='model')
    output.sort_values(by='mean_test_acc_tuned', ascending=False, inplace=True)
    
    return output #baseline_output, tuned_output


#output = BaseLineModels(models_to_pass, scaler.fit_transform(X_train), y_train)