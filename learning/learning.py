import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from joblib import dump
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold  
from sklearn.linear_model import LogisticRegression
import mlflow
from sklearn.feature_selection import SelectKBest
from utils.utils import load_data,load_model

mlflow.autolog()
model_path = 'model/model_LR.joblib'
scaler_path = 'model/scaler.joblib'
encoder_X_path = 'model/encoder_X.joblib'
encoder_Y_path = 'model/encoder_Y.joblib'
selectBeast_path = 'model/select_beast.joblib'

X, y = load_data('data/test2.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.20, random_state=42, stratify=y)

le1 = LabelEncoder()
X_train['handedness.label'] = le1.fit_transform(X_train['handedness.label'])
X_test['handedness.label'] = le1.transform(X_test['handedness.label'])
dump(le1,encoder_X_path)

X_train = X_train.astype("float32")

le2 = LabelEncoder()
y_train = le2.fit_transform(y_train)
y_test = le2.transform(y_test)
dump(le2,encoder_Y_path)

k = 48  
selector = SelectKBest(k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
dump(selector,selectBeast_path)



md = LogisticRegression(C=100,penalty='l1',solver='liblinear')
scaler = StandardScaler()
X_train_selected = scaler.fit_transform(X_train_selected)
X_test_selected = scaler.transform(X_test_selected)
dump(scaler, scaler_path)


md.fit(X_train_selected,y_train)
pred=md.predict(X_test_selected)
print(f1_score(y_test, pred,average='micro'))
dump(md, model_path)
       

