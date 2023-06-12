import numpy as np
import pandas as pd
from  joblib import load
from sklearn.preprocessing import LabelEncoder
def load_data(df:pd.DataFrame):
    df = df[df.columns.drop(list(df.filter(regex='world_')))]
    df = df.drop([col for col in df.columns if col.startswith('Unnamed')], axis=1)
    X = df.drop(['handedness.score'], axis=1)
    return X
def load_model(model_path):
    model = load(model_path)
    return model

def predict(model, X):
    predictions = model.predict(X)
    return predictions
def preprocess_data(X_test):
    le = LabelEncoder()

    X_test['handedness.label'] = le.fit_transform(X_test['handedness.label'])

    X_test = X_test.astype("float32")

    return  X_test
def perform_processing(
        data: pd.DataFrame
) -> pd.DataFrame:
    # NOTE(MF): sample code
    # preprocessed_data = preprocess_data(data)
    # models = load_models()  # or load one model
    # please note, that the predicted data should be a proper pd.DataFrame with column names
    # predicted_data = predict(models, preprocessed_data)
    # return predicted_data

    # for the simplest approach generate a random DataFrame with proper column names and size

    result = final_predict(data)

    predicted_data = pd.DataFrame(
        result,
        columns=['letter']
    )

    print(f'{predicted_data=}')

    return predicted_data
def preprocess_data(X_test):
    le = LabelEncoder()

    X_test['handedness.label'] = le.fit_transform(X_test['handedness.label'])

    X_test = X_test.astype("float32")

    return  X_test

def final_predict(df:pd.DataFrame):
    model_path = 'model/model_LR.joblib'
    scaler_path = 'model/scaler.joblib'
    encoder_X_path = 'model/encoder_X.joblib'
    encoder_Y_path = 'model/encoder_Y.joblib'
    selectBeast_path = 'model/select_beast.joblib'
    scaler = load(scaler_path)
    leX = load(encoder_X_path)
    leY = load(encoder_Y_path)
    selectBeast = load(selectBeast_path)

    
    X_test = load_data(df)
    X_test  = preprocess_data(X_test)
    X_test=selectBeast.transform(X_test)
    X_test=scaler.transform(X_test)
    
    model = load_model(model_path)

    predictions = predict(model, X_test)
    predictions = leY.inverse_transform(predictions)

    return predictions
