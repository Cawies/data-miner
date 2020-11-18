# External libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Internal modules
from processing.data_management import load_dataset_locally
from config import config
import pipelines as pipelines
from modelling import modelling


def run_package():
    raw_data = load_dataset_locally(file_name=config.DATA_FILE)
    processed_data = pipelines.cleaning_pipeline.fit_transform(raw_data)
    reduced_data = pipelines.reduction_pipeline.fit_transform(processed_data)

    #X_train, X_test, y_train, y_test = train_test_split(
    #    data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    #)  # we are setting the seed here

    X_train, X_test, y_train, y_test = train_test_split(
        reduced_data.drop(config.TARGET, axis=1), 
        reduced_data[config.TARGET], 
        test_size=0.20, 
        stratify=reduced_data[config.TARGET], 
        random_state=config.RANDOM_STATE)
    
    scaler = StandardScaler()
    performance_output = modelling.BaseLineModels(config.MODELS, scaler.fit_transform(X_train), y_train)
    performance_output.to_excel('model_performances.xlsx', index=False)
    



if __name__ == '__main__':
    run_package()