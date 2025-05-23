import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['label'])
    y = pd.get_dummies(df['label']).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y