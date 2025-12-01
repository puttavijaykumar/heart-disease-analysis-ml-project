import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_and_encode(df):
    df = df.drop(['id', 'dataset'], axis=1).copy()

    # Encode categorical columns
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['cp'] = df['cp'].map({'typical angina': 1, 'atypical angina': 2, 'non-anginal': 3, 'asymptomatic': 4})
    df['fbs'] = df['fbs'].map({True: 1, False: 0})
    df['restecg'] = df['restecg'].map({'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2})
    df['exang'] = df['exang'].map({True: 1, False: 0})
    df['slope'] = df['slope'].map({'upsloping': 1, 'flat': 2, 'downsloping': 3})
    df['thal'] = df['thal'].map({'normal': 1, 'fixed defect': 2, 'reversable defect': 3})

    # Handle missing values
    num_cols = ['age','trestbps','chol','thalch','oldpeak']
    cat_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df

def apply_smote(df):
    X = df.drop('num', axis=1)
    y = df['num']

    categorical_idx = [5,6,7,8,9,10,11,12]

    smote_nc = SMOTENC(categorical_features=categorical_idx, random_state=42)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)

    return X_resampled, y_resampled
    
def scale_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test
