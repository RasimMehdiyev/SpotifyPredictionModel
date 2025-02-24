import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import math
from scipy.sparse import issparse

def read_data():
    data = pd.read_csv(r'C:\Users\rasim\Downloads\archive\spotify_songs.csv')
    if 'track_popularity' not in data.columns:
        print("Column 'track_popularity' does not exist in the dataset.")
        exit(1)

    # print name of columns
    # print(data.columns)
    return data

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b, lambda_=1):
    m, n = X.shape
    Z = np.dot(X, w) + b
    F_wb = sigmoid(Z)
    loss_i = (-y * np.log(F_wb)) - (1 - y) * np.log(1 - F_wb)
    total_cost = np.sum(loss_i) / m
    return total_cost

def compute_gradient(X, y, w, b, lambda_=None):
    m, n = X.shape
    Z = X.dot(w) + b if issparse(X) else np.dot(X, w) + b
    F_wb = sigmoid(Z)
    dj_dw = X.T.dot(F_wb - y) / m if issparse(X) else np.dot(X.T, (F_wb - y)) / m
    dj_db = np.sum(F_wb - y) / m
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    J_history = []
    w_history = []
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)
        w_in -= alpha * dj_dw
        b_in -= alpha * dj_db
        if i < 100000:
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
    return w_in, b_in, J_history, w_history

def predict(X, w, b):
    Z = X.dot(w) + b if issparse(X) else np.dot(X, w) + b
    F_wb = sigmoid(Z)
    p = np.where(F_wb > 0.5, 1, 0)
    return p

def main():
    data = read_data()
    
    categorical_features = ['track_id', 'track_name', 'track_artist', 'track_album_id', 'track_album_name', 'playlist_name', 'playlist_id', 'playlist_genre', 'playlist_subgenre', 'key', 'mode']  # list all categorical features
    numerical_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']  # list all numerical features


    # print("Columns in DataFrame:", data.columns)

    X = data.drop(columns=['track_popularity'])  # Features
    y = data['track_popularity']  # Target

    for col in categorical_features:
        if col not in X.columns:
            print(f"Column '{col}' does not exist in the dataset.")
            exit(1)

    for col in numerical_features:
        if col not in X.columns:
            print(f"Column '{col}' does not exist in the dataset.")
            exit(1)

    # print("Columns in X after dropping 'track_popularity':", X.columns)
    # print("Categorical features:", categorical_features)
    # print("Numerical features:", numerical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse=True), categorical_features)])  # Keep output sparse

    try:
        X = preprocessor.fit_transform(X)
    except ValueError as e:
        print("Error in preprocessing:", e)
        print('\n')
        return
    
    if isinstance(X, np.ndarray):
        print("X is already a dense matrix.\n")
    else:
        print("Converting X to a dense matrix.\n")
        X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    w = np.zeros((X_train.shape[1], 1))
    b = 0

    # Train the model
    alpha = 0.01
    num_iters = 1000
    w, b, J_history, w_history = gradient_descent(X_train, y_train, w, b, compute_cost, compute_gradient, alpha, num_iters, None)

    # Predictions
    predictions = predict(X_test, w, b)
    

if __name__ == '__main__':
    main()
