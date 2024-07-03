import pickle
import pandas as pd

def predict_duration(df, categorical, model_file):
    with open(model_file, 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    return y_pred