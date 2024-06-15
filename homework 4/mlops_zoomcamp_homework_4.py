import pickle
import pandas as pd
import numpy as np
import argparse

def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].fillna(-1).astype('int').astype('str')
    return df

def predict_duration(model, dv, df):
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred

def main(year, month):
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    df = read_data(filename)
    y_pred = predict_duration(model, dv, df)

    std_dev = np.std(y_pred)
    print(f'The standard deviation of the estimated duration is: {std_dev:.2f}')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predictions': y_pred})

    output_file = f'output_file_{year:04d}-{month:02d}.parquet'
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)

    print(f'The estimated average duration is: {y_pred.mean():.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict trip duration')
    parser.add_argument('--year', type=int, required=True, help='Year of the data')
    parser.add_argument('--month', type=int, required=True, help='Month of the data')
    args = parser.parse_args()
    main(args.year, args.month)
    
#python mlops_zoomcamp_homework_4.py --year 2023 --month 4
#python mlops_zoomcamp_homework_4.py --year 2023 --month 5
