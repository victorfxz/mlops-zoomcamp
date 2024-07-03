import pandas as pd
from data_processing import prepare_data

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    expected_data = [
        (1, 1, 8.0),
        (1, '1', 8.0),
        (1, '1', 59.0),
    ]

    expected_columns = ['PULocationID', 'DOLocationID', 'duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)

    result_df = prepare_data(df, categorical)
    assert result_df.to_dict(orient='records') == expected_df.to_dict(orient='records')

test_prepare_data()