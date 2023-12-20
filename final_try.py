import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold


def load_data():
    data = pd.read_csv('./spotify_songs.csv')
    return data

def separate_data(data):
    popularity_bins = [0, 33, 66, 100]  # Adjust bins as needed
    popularity_labels = [0, 1, 2]
    data = data.drop(columns=['track_id', 'track_name', 'track_album_id', 'track_artist', 
                            'playlist_name', 'playlist_genre', 'playlist_subgenre', 
                            'track_album_name', 'track_album_release_date', 'playlist_id'])
    data['track_popularity'] = pd.cut(data['track_popularity'], bins=popularity_bins, labels=popularity_labels, include_lowest=True, right=True)

    sampled_data = pd.DataFrame()
    for label in popularity_labels:
        class_samples = data[data['track_popularity'] == label]
    # Check if the class has enough samples
        if len(class_samples) >= 6000:
            class_samples = class_samples.sample(n=6000, random_state=42)
        sampled_data = pd.concat([sampled_data, class_samples])
    sampled_data = sampled_data.reset_index(drop=True)
    data = sampled_data
    X = data.drop(columns=['track_popularity'])
    y = data['track_popularity']


    return X, y

def standardize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

class SoftmaxRegression:
    def __init__(self, learning_rate=0.05, num_iterations=3000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def cross_validate(self, X_train, y_train, X_val, y_val, learning_rates, num_iterations_list):
        best_f1 = 0
        best_params = {'learning_rate': None, 'num_iterations': None}

        for learning_rate in learning_rates:
            for num_iterations in num_iterations_list:
                print(f"Testing learning rate {learning_rate}, iteration {num_iterations}")

                self.__init__(learning_rate, num_iterations)
                self.fit(X_train, y_train)

                y_pred = self.predict(X_val)
                f1 = f1_score(y_val, y_pred, average='weighted')

                if f1 > best_f1:
                    best_f1 = f1
                    best_params['learning_rate'] = learning_rate
                    best_params['num_iterations'] = num_iterations

                print(f"  - F1 Score: {f1}, Best F1: {best_f1}")

        return best_params

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.k = len(self.classes)
        self.m, self.n = X.shape
        self.W = np.zeros((self.n, self.k))
        self.b = np.zeros(self.k)
        self.accuracy_history = []
        self.f1_history = []

        y_encoded = self.one_hot_encode(y)

        for _ in range(self.num_iterations):
            scores = np.dot(X, self.W) + self.b
            probabilities = softmax(scores)
            gradients_w = np.dot(X.T, (probabilities - y_encoded)) / self.m
            gradients_b = np.mean(probabilities - y_encoded, axis=0)
            
            self.W -= self.learning_rate * gradients_w
            self.b -= self.learning_rate * gradients_b

            y_pred = self.predict(X)
            accuracy = np.sum(y_pred == y) / self.m
            f1 = f1_score(y, y_pred, average='weighted')
            self.accuracy_history.append(accuracy)
            self.f1_history.append(f1)

    def predict(self, X):
        scores = np.dot(X, self.W) + self.b
        probabilities = softmax(scores)
        return np.argmax(probabilities, axis=1)

    def one_hot_encode(self, y):
        y_encoded = np.zeros((y.size, self.k))
        for i, label in enumerate(y):
            y_encoded[i, self.classes == label] = 1
        return y_encoded
    
    def predict_proba(self, X):
        scores = np.dot(X, self.W) + self.b
        probabilities = softmax(scores)
        return probabilities

    def split_data(self, X, y, train_size=0.6, test_size=0.2, random_state=42):
    # First split: 60% training, 40% temporary (test + validation)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=train_size, random_state=random_state
        )

        # Adjust test_size for the second split: it's 50% of the remaining data
        temp_test_size = test_size / (1 - train_size)

        # Second split: split the temporary set into test and validation
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=temp_test_size, random_state=random_state
        )

        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def plot_performance_over_iterations(self, accuracy_history, f1_history, file_name):
        plt.figure(figsize=(10, 6))
        plt.plot(accuracy_history, label='Accuracy')
        plt.plot(f1_history, label='F1 Score')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Model Performance Over Iterations')
        plt.legend()
        plt.savefig(file_name)
    
    def plot_roc_curves(self, y_test, predicted_probabilities, classes, file_name):
        y_test_binarized = label_binarize(y_test, classes=classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], predicted_probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 7))
        for i in range(len(classes)):
            plt.plot(fpr[i], tpr[i], label=f'Class {classes[i]} (area = {roc_auc[i]:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Class')
        plt.legend(loc='best')
        plt.savefig(file_name)

    def plot_feature_weights(self, weights, feature_names, classes, file_name):
        for i, class_name in enumerate(classes):
            plt.figure(figsize=(12, 8))
            plt.barh(feature_names, weights[:, i])
            plt.xlabel('Weight')
            plt.title(f'Feature Weights for Class {class_name}')
            plt.savefig(f'{file_name}_class_{class_name}.png')

    def plot_confusion_matrix(self, y_true, y_pred, class_names,file_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(file_name)

    def plot_violin(self, y_true, y_pred, class_names, file_name):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

        # Map predicted classes to class names
        class_map = {i: name for i, name in enumerate(class_names)}
        y_pred_names = [class_map[pred] for pred in y_pred]

        # Create a DataFrame for seaborn
        data = pd.DataFrame({
            'True Class': y_true,
            'Predicted Class': y_pred,
            'Predicted Class Name': y_pred_names
        })

        plt.figure(figsize=(10, 7))
        sns.violinplot(x='True Class', y='Predicted Class', hue='Predicted Class Name', data=data)
        plt.title("Violin Plot")
        plt.xlabel("True Class")
        plt.ylabel("Predicted Class")
        plt.savefig(file_name)

    def plot_heatmap(self, y_true, y_pred, class_names, file_name):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Convert confusion matrix to DataFrame for better visualization
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        plt.figure(figsize=(10, 7))
        # Create the heatmap
        sns.heatmap(cm_df, annot=True, fmt='g', cmap='viridis')
        plt.title("Confusion Matrix Heatmap")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.savefig(file_name)

def main():
    data = load_data()
    X, y = separate_data(data)
    X_scaled = standardize_data(X)

    model = SoftmaxRegression()

    X_train, X_test, X_val, y_train, y_test, y_val = model.split_data(X_scaled, y)

    learning_rates = [0.01, 0.05, 0.1]
    num_iterations_list = [1000, 2000, 3000]

    

    # Cross-validate to find the best hyperparameters
    # best_params = model.cross_validate(X_train, y_train, X_val, y_val, learning_rates, num_iterations_list)

    # print("Best hyperparameters:", best_params)

    # model = SoftmaxRegression(learning_rate=best_params['learning_rate'], num_iterations=best_params['num_iterations'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.sum(y_pred == y_test)/y_test.size


    # Print results

    print("Accuracy:", accuracy*100)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1 Score:", f1*100)

    # Plots 
    model.plot_confusion_matrix(y_test, y_pred, model.classes, 'softmax_confusion_matrix.png')
    model.plot_violin(y_test, y_pred, model.classes, 'softmax_violin.png')
    model.plot_heatmap(y_test, y_pred, model.classes, 'softmax_heatmap.png')
    predicted_probabilities = model.predict_proba(X_test)
    model.plot_roc_curves(y_test, predicted_probabilities, model.classes, 'softmax_roc_curves.png')
    model.accuracy_history = np.array(model.accuracy_history)
    model.f1_history = np.array(model.f1_history)
    model.plot_performance_over_iterations(model.accuracy_history, model.f1_history, 'performance_over_iterations.png')
    feature_names = X.columns
    model.plot_feature_weights(model.W, feature_names, model.classes, 'feature_weights')



if __name__ == "__main__":
    main()
