import os
import sys
import pickle
import pandas as pd

def get_s3_options():
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    if s3_endpoint_url:
        return {'client_kwargs': {'endpoint_url': s3_endpoint_url}}
    return None

def read_data(filename, categorical_columns=None):
    storage_options = get_s3_options()
    df = pd.read_parquet(filename, storage_options=storage_options)
    
    if categorical_columns:
        df = prepare_data(df, categorical_columns)
    
    return df

def save_data(filename, df):
    storage_options = get_s3_options()
    df.to_parquet(filename, engine='pyarrow', compression=None, index=False, storage_options=storage_options)

def prepare_data(df, categorical_columns):
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical_columns] = df[categorical_columns].fillna(-1).astype(int).astype(str)
    return df

def get_path(template, year, month):
    return template.format(year=year, month=month)

def main(year, month):
    input_template = os.getenv('INPUT_FILE_PATTERN', 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    output_template = os.getenv('OUTPUT_FILE_PATTERN', 's3://nyc-duration/out/year={year:04d}/month={month:02d}/predictions.parquet')
    
    input_file = get_path(input_template, year, month)
    output_file = get_path(output_template, year, month)
    categorical_columns = ['PULocationID', 'DOLocationID']
    
    with open('model.bin', 'rb') as model_file:
        dv, lr = pickle.load(model_file)
    
    df = read_data(input_file, categorical_columns)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype(str)
    
    X_val = dv.transform(df[categorical_columns].to_dict(orient='records'))
    y_pred = lr.predict(X_val)
    
    print('Predicted mean duration:', y_pred.mean())
    
    df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predicted_duration': y_pred})
    save_data(output_file, df_result)

if __name__ == "__main__":
    year, month = map(int, sys.argv[1:3])
    main(year, month)
