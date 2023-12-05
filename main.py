import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.sparse import issparse
import statsmodels.api as sm
import plots as pl

def load_test_data():
    data = np.loadtxt("./ex2data1.txt", delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y

def load_data():
    data = pd.read_csv('./spotify_songs.csv')
    print(type(data))
    # use only 100 songs
    data = data[:15000]
    # print(data)
    return data


def compute_gradient(X, y, w, b,lambda_=1):
    m,n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    # Compute the predicted value
    Z = np.dot(X, w) + b
    F_wb = sigmoid(Z)
    
    # Ensure F_wb and y have the correct shape
    F_wb = F_wb.reshape(-1, 1)
    if isinstance(y, pd.Series):
        y = y.values.reshape(-1, 1)
    elif isinstance(y, np.ndarray) and y.ndim == 1:
        y = y.reshape(-1, 1)

    # Compute the gradients
    error = F_wb - y
    dj_dw = np.dot(X.T, error) / m
    dj_db = np.sum(error) / m

    reg_term_dw = (lambda_ / m) * w
    dj_dw += reg_term_dw

    return dj_dw, dj_db

def gradient_descent(X,y,w_in,b_in,cost_function,gradient_function,alpha,num_epochs,lambda_=1):
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    for epoch in range(num_epochs):
        for i in range(m):

            # Calculate the gradient and update the parameters
            index = np.random.randint(0, m)
            X = X[index:index+1].reshape(1, -1)
            y = y[index:index +1].reshape(1, -1)

            
            dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

            # Update Parameters using w, b, alpha and gradient
            w_in = w_in - alpha * dj_dw    
            # print(b_in)
            b_in = b_in - alpha * dj_db  

            # b_in = b_in[0][0]
            b_in = b_in.reshape(-1,1)
            # turn into a scalar
            b_in = b_in[0][0]
            

            # Save cost J at each iteration
        cost =  cost_function(X, y, w_in, b_in, lambda_)
                # print("cost.shape:",cost.shape)
        J_history.append(cost)
        if epoch % 100 == 0:
            cost = cost_function(X, y, w, b, lambda_)
            print(f"Epoch {epoch}: Cost {cost:.4f}")

            # Print cost every at intervals 10 times or as many iterations if < 10
            # if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            #     w_history.append(w_in)
        # print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
    return w_in, b_in, J_history, w_history

def separate_data(data):
    # Separate the data into X and y
    X = data.drop(columns=['track_id', 'track_name', 'track_album_id', 'track_artist', 'playlist_name',
                           'playlist_genre', 'playlist_subgenre', 'key', 'mode', 'track_popularity',
                           'track_album_name', 'track_album_release_date', 'playlist_id','loudness','duration_ms','valence'])    # drop non-numeric columns
    y = data['track_popularity']    # keep only the 'track_popularity' column

    # print("X.shape",X.shape)
    # print("y.shape",y.shape)
    return X,y

def count_popularity_categories(y):
    pop_list = []
    # 0 -> not popular
    # 1 -> moderately popular
    # 2 -> popular

    # count all 3
    pop_list.append(np.count_nonzero(y == 0))
    pop_list.append(np.count_nonzero(y == 1))
    pop_list.append(np.count_nonzero(y == 2))

    return pop_list
        
def categorize_popularity(track_popularity, low_threshold=25, high_threshold=75):
    categories = np.zeros_like(track_popularity, dtype=int)
    categories[track_popularity > high_threshold] = 2  # Popular
    categories[(track_popularity > low_threshold) & (track_popularity <= high_threshold)] = 1  # Somewhat popular
    # Not popular will remain as 0
    return categories

def modify_target_for_multiclass(y, class_label):
    return np.where(y == class_label, 1, 0)

def train_multiclass(X, y, num_classes, base_alpha, num_iters):
    models = []
    for class_label in range(num_classes):
        y_mod = modify_target_for_multiclass(y, class_label)
        w = np.random.randn(X.shape[1], 1) * 0.01
        b = 0.

        for iter in range(num_iters):
            alpha = dynamic_alpha(iter, base_alpha)
            w, b, _, _ = gradient_descent(X, y_mod, w, b, compute_cost, compute_gradient, alpha, 1)
        
        models.append((w, b))
    
    return models


def predict_multiclass(X, models):
    predictions = [sigmoid(np.dot(X, w) + b) for w, b in models]
    return np.argmax(np.hstack(predictions), axis=1)

def sigmoid(z):
    # clip 
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, w, b,lambda_=1):
    m,n = X.shape

    Z = np.dot(X,w) + b

    F_wb = sigmoid(Z)
    # Ensure F_wb and y have the correct shape
    F_wb = F_wb.reshape(-1, 1)
    if isinstance(y, pd.Series):
        y = y.values.reshape(-1, 1)
    elif isinstance(y, np.ndarray) and y.ndim == 1:
        y = y.reshape(-1, 1)

    # Compute the loss for each example
    loss_i = (-y * np.log(F_wb + 1e-7)) - ((1 - y) * np.log(1 - F_wb + 1e-7))
    reg_term = (lambda_ / (2 * m)) * np.sum(np.square(w))
    
    # Sum the loss over all examples to get the total cost
    total_cost = (np.sum(loss_i) / m) + reg_term
    return total_cost.item()  # Convert the numpy scalar to a Python scalar

def compute_gradient(X, y, w, b,lambda_=None):
    m,n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    # Compute the predicted value
    Z = np.dot(X, w) + b
    F_wb = sigmoid(Z)
    
    # Ensure F_wb and y have the correct shape
    F_wb = F_wb.reshape(-1, 1)
    if isinstance(y, pd.Series):
        y = y.values.reshape(-1, 1)
    elif isinstance(y, np.ndarray) and y.ndim == 1:
        y = y.reshape(-1, 1)

    # Compute the gradients
    error = F_wb - y
    dj_dw = np.dot(X.T, error) / m
    dj_db = np.sum(error) / m

    return dj_dw, dj_db

def gradient_descent(X,y,w_in,b_in,cost_function,gradient_function,alpha,num_iters,lambda_):
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw    
        # print(b_in)
        b_in = b_in - alpha * dj_db  

        # b_in = b_in[0][0]
        b_in = b_in.reshape(-1,1)
        # turn into a scalar
        b_in = b_in[0][0]
          

        # Save cost J at each iteration
        cost =  cost_function(X, y, w_in, b_in, lambda_)
            # print("cost.shape:",cost.shape)
        J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    # pl.plot_cost_history(J_history)

    return w_in, b_in, J_history, w_history

def dynamic_alpha(iteration, base_alpha, decay_rate=0.9):
    return base_alpha * (decay_rate ** iteration)

def separate_data(data):
    # Separate the data into X and y
    X = data.drop(columns=['track_id', 'track_name', 'track_album_id', 'track_artist', 'playlist_name',
                           'playlist_genre', 'playlist_subgenre', 'key', 'mode', 'track_popularity',
                           'track_album_name', 'track_album_release_date', 'playlist_id','loudness','duration_ms','valence'])    # drop non-numeric columns
    y = data['track_popularity']    # keep only the 'track_popularity' column

    # print("X.shape",X.shape)
    # print("y.shape",y.shape)
    return X,y

def count_popularity_categories(y):
    pop_list = []
    # 0 -> not popular
    # 1 -> moderately popular
    # 2 -> popular

    # count all 3
    pop_list.append(np.count_nonzero(y == 0))
    pop_list.append(np.count_nonzero(y == 1))
    pop_list.append(np.count_nonzero(y == 2))

    return pop_list
        



def categorize_popularity(track_popularity, low_threshold=25, high_threshold=75):
    categories = np.zeros_like(track_popularity, dtype=int)
    categories[track_popularity > high_threshold] = 2  # Popular
    categories[(track_popularity > low_threshold) & (track_popularity <= high_threshold)] = 1  # Somewhat popular
    # Not popular will remain as 0
    return categories

def modify_target_for_multiclass(y, class_label):
    return np.where(y == class_label, 1, 0)

def train_multiclass(X, y, num_classes, alpha, num_iters,lambda_):
    models = []
    for class_label in range(num_classes):
        y_mod = modify_target_for_multiclass(y, class_label)
        w = np.random.randn(X.shape[1], 1) * 0.01
        b = 0.

        w, b, J_history, w_history = gradient_descent(X ,y_mod, w, b, compute_cost, compute_gradient, alpha, num_iters, lambda_)
        models.append((w, b))

    
    return models


def predict_multiclass(X, models):
    predictions = np.array([sigmoid(np.dot(X, w) + b) for w, b in models]).T
    return np.argmax(np.hstack(predictions), axis=1)



def main():
    # load data
    data = load_data()
    X,y = separate_data(data)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    y  = categorize_popularity(y)
    # y = y.reshape(-1,1)
    # y_scaled = scaler.fit_transform(y)
    # y = pd.DataFrame(y_scaled, columns=y.columns)



    numerical_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'tempo']
    for col in numerical_features:
        if col not in X.columns:
            print(f"Column '{col}' does not exist in the dataset.")
            return

    lambda_ = 0.1
    num_classes = 3
    alpha = 0.001 # TRY TO DECREASE THE LEARNING RATE
    num_iters = 1000

    models = train_multiclass(X, y, num_classes, alpha, num_iters, lambda_)
    predictions = predict_multiclass(X, models)
    print("Accuracy manual:", accuracy_score(y, predictions)*100, "%")


    # using library
    model = LogisticRegression(multi_class='ovr')
    model.fit(X, y)
    prediction_1 = model.predict(X)

    # print("Original multi-class y:\n",y)
    # print("Predictions:\n",predictions)

    print("Accuracy library:", accuracy_score(y, prediction_1)*100, "%")
    print("Confusion Matrix:\n", confusion_matrix(y, predictions))
    # print("\nClassification Report:\n", classification_report(y, predictions))

    print(count_popularity_categories(y))
    print(count_popularity_categories(predictions))

    # ACCURACY VS ITERATION PLOT 

    # plots

    pl.plot_predictions(X, y, predictions)
    # print(data.columns)
    return True

if __name__ == "__main__":
    main()
