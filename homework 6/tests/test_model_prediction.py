import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from model_prediction import predict_duration

def test_predict_duration():
    data = [
        {'PULocationID': 1, 'DOLocationID': 1, 'duration': 8.0},
        {'PULocationID': 1, 'DOLocationID': '1', 'duration': 8.0},
        {'PULocationID': 1, 'DOLocationID': '1', 'duration': 59.0},
    ]

    df = pd.DataFrame(data)

    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(df[['PULocationID', 'DOLocationID']].to_dict(orient='records'))
    y = df['duration'].values

    lr = LinearRegression()
    lr.fit(X, y)

    with open('test_model.bin', 'wb') as f:
        pickle.dump((dv, lr), f)

    y_pred = predict_duration(df, ['PULocationID', 'DOLocationID'], 'test_model.bin')

    np.testing.assert_allclose(y_pred, lr.predict(X), atol=1e-5)

    os.remove('test_model.bin')

test_predict_duration()