# Internal modules
from processing.data_management import load_dataset_locally
from config import config
import pipelines as pipelines


def run_package():
    raw_data = load_dataset_locally(file_name=config.DATA_FILE)
    processed_data = pipelines.cleaning_pipeline.fit_transform(raw_data)
    reduced_data = pipelines.reduction_pipeline.fit_transform(processed_data)
    



if __name__ == '__main__':
    run_package()