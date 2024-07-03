import sys
import pandas as pd
from data_processing import prepare_data
from model_prediction import predict_duration

def main(year, month, categorical, input_file, output_file, model_file):
    df = pd.read_parquet(input_file)
    df = prepare_data(df, categorical)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    y_pred = predict_duration(df, categorical, model_file)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)

if __name__ == '__main__':
    year = 2024
    month = 3
    categorical = ['PULocationID', 'DOLocationID']
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'
    model_file = 'model.bin'
    main(year, month, categorical, input_file, output_file, model_file)