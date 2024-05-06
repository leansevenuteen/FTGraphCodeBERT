from GraphCodeBERT import gen_data
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
import pandas as pd
import numpy as np

d_path = 'Data/AST'
train_df = gen_data(f"{d_path}/train")
test_df = gen_data(f"{d_path}/test")

def split_data(data, d_type, g_type):
    if d_type == 'train':
        num = 9600 if g_type != 'PDG' else (9600-22)
        labels = [1 if idx < num else 0 for idx in range(data.shape[0])]
    elif d_type == 'test':
        num = 1500 if g_type != 'PDG' else (1500-3)
        labels = [1 if idx < num else 0 for idx in range(data.shape[0])]

    indices = np.random.permutation(data.shape[0])
    shuffled_data = data.iloc[indices]
    shuffled_labels = np.array(labels)[indices]

    return shuffled_data, shuffled_labels

X_train, y_train = split_data(train_df, 'train', 'AST')
X_test, y_test = split_data(test_df, 'test', 'AST')

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(classification_report(y_test, rf_pred))

xg = xgb.XGBClassifier(n_estimators=100, max_depth=1, random_state=42, learning_rate=1.0)
xg.fit(X_train, y_train)
xg_pred = xg.predict(X_test)
print(classification_report(y_test, xg_pred))

ada = AdaBoostClassifier(n_estimators=100, random_state=0)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
print(classification_report(y_test, ada_pred))