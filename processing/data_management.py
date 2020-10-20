# External libraries
import pandas as pd

# Internal modules
from config import config

def load_dataset_locally(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.DATA_DIR}/{file_name}")
    return _data