
import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:

    data = pd.read_csv('/home/src/mlops/Youtube/youtube.csv')
    
    return data

