# Internal modules
from processing.data_management import load_dataset_locally
from config import config


def run_package():
    data = load_dataset_locally(file_name=config.DATA_FILE)
    print(data.columns)


if __name__ == '__main__':
    run_package()