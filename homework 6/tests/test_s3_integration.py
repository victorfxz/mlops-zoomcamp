import boto3
import moto
import pandas as pd
import os

@moto.mock_s3
def test_s3_integration():
    data = [
        (1, 1, 8.0),
        (1, '1', 8.0),
        (1, '1', 59.0),
    ]

    columns = ['PULocationID', 'DOLocationID', 'duration']
    df = pd.DataFrame(data, columns=columns)

    s3_client = boto3.client('s3', region_name='us-east-1')
    s3_client.create_bucket(Bucket='test-bucket')

    output_file = 'test_data.parquet'
    df.to_parquet(output_file, engine='pyarrow', index=False)

    with open(output_file, 'rb') as f:
        s3_client.upload_fileobj(f, 'test-bucket', output_file)

    s3_resource = boto3.resource('s3', region_name='us-east-1')
    obj = s3_resource.Object('test-bucket', output_file)
    body = obj.get()['Body'].read()

    expected_data = [
        (1, 1, 8.0),
        (1, '1', 8.0),
        (1, '1', 59.0),
    ]

    expected_columns = ['PULocationID', 'DOLocationID', 'duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)

    result_df = pd.read_parquet(body, engine='pyarrow')
    assert result_df.to_dict(orient='records') == expected_df.to_dict(orient='records')

    os.remove(output_file)
    s3_client.delete_object(Bucket='test-bucket', Key=output_file)
    s3_client.delete_bucket(Bucket='test-bucket')

test_s3_integration()