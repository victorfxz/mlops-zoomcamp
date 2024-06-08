import requests
import pandas as pd
from io import BytesIO
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from typing import Dict
import mlflow

# Verifica se os decoradores estão disponíveis
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_loader
def ingest_files(*args, **kwargs) -> pd.DataFrame:
    """Ingest data from a URL and return a pandas dataframe"""
    url = 'https://github.com/victorfxz/datasets/raw/main/yellow_tripdata_2023-03.parquet'
    response = requests.get(url)
    
    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f'Failed to fetch data from {url}. {e}')
    
    df = pd.read_parquet(BytesIO(response.content))
    return df

@transformer
def read_dataframe(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """Read and preprocess the dataframe"""
    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    categorical_cols = ['PULocationID', 'DOLocationID']
    df[categorical_cols] = df[categorical_cols].astype(str)
    
    return df

@transformer
def train_model(df: pd.DataFrame, *args, **kwargs) -> Dict[str, object]:
    """Train a linear regression model on the preprocessed data"""
    # Create features
    dict_list = df[['PULocationID', 'DOLocationID']].apply(lambda row: row.to_dict(), axis=1).tolist()
    
    # Fit DictVectorizer
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(dict_list)
    y = df['duration']
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Print intercept
    print(f"Intercept: {model.intercept_}")
    
    return {'vectorizer': vectorizer, 'model': model}

@data_exporter
def register_model(model_artifacts: Dict[str, object], *args, **kwargs) -> None:
    """Register the model and artifacts with MLflow"""
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("yellow_taxi_regression")
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        mlflow.log_metric("intercept", model_artifacts['model'].intercept_)
        mlflow.sklearn.log_model(model_artifacts['model'], "model")
        mlflow.sklearn.log_model(model_artifacts['vectorizer'], "vectorizer")