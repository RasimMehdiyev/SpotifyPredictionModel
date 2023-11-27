import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cost_history(J_history):
    plt.plot(J_history)
    plt.xlabel('Iteration')
    plt.ylabel('$J(\mathbf{w},b)$')
    plt.title('Cost function using Gradient Descent')
    plt.show()

def plot_results(X, y, w, b):
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, s=25)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Scatter plot of training data')
    plt.show()

    # Plot the decision boundary
    # Only need 2 points to define a line, so choose two endpoints
    x1 = np.array([np.min(X[:,0]), np.max(X[:,0])])
    x2 = -(b + w[0]*x1) / w[1]

    # Plot the decision boundary
    plt.plot(x1, x2, c='r', label='Decision boundary')
    plt.legend()
    plt.show()

def plot_categorical_results(X, y):
    numerical_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo']
    for col in numerical_features:
        if col not in X.columns:
            print(f"Column '{col}' does not exist in the dataset.")
            return
        
    plt.scatter(X['danceability'], X['energy'], c=y, cmap=plt.cm.coolwarm, s=25)
    plt.xlabel('danceability')
    plt.ylabel('energy')
    plt.title('Scatter plot of training data')
    plt.show()


    # Plot the decision boundary
def plot_predictions(X, y, predictions):
    numerical_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo']
    for col in numerical_features:
        if col not in X.columns:
            print(f"Column '{col}' does not exist in the dataset.")
            return
        
    figures, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Scatter plot of actual vs predicted classes
    axes[0].scatter(range(len(y)), y, color='blue', label='Actual', alpha=0.5)
    axes[0].scatter(range(len(predictions)), predictions, color='red', label='Predicted', alpha=0.5)
    axes[0].set_title('Actual vs Predicted Classes')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Class')
    axes[0].legend()
    axes[0].grid(True)

    # Bar plot of the distribution of actual vs predicted classes
    df_comparison = pd.DataFrame({'Actual': y, 'Predicted': predictions})

    # Count occurrences
    class_counts = df_comparison.melt(var_name='Type', value_name='Class').groupby(['Type', 'Class']).size().reset_index(name='Count')

    # Create bar plot
    sns.barplot(x='Class', y='Count', hue='Type', data=class_counts, ax=axes[1])
    axes[1].set_title('Distribution of Actual vs Predicted Classes')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].grid(True)

    plt.show()


