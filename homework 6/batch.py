import os
import sys
import pickle
import pandas as pd

def read_data(filename, categorical=None):
    '''
    This function reads the parquet data and prepares it
    '''
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    options = None
    if S3_ENDPOINT_URL:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }       

    df = pd.read_parquet(filename, storage_options=options)
    if categorical:
        df = prepare_data(df, categorical)
    
    return df

def save_data(filename, df):
    '''
    This function saves the parquet data
    '''
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    options = None
    if S3_ENDPOINT_URL:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }  

    df.to_parquet(
        filename,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def prepare_data(df, categorical):
    '''
    This function massages the data by removing outliers and reformatting
    '''
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def get_input_path(year, month):
    '''
    Gets input path name where year and month are inputs
    '''
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    '''
    Gets output path name where year and month are inputs
    '''
    default_output_pattern = 's3://nyc-duration/out/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def main(year, month):
    '''
    Main function that does all the reading of the data, prediction, and formatting
    '''
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    categorical = ['PULocationID', 'DOLocationID']

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(filename=output_file, df=df_result)

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year=year, month=month)