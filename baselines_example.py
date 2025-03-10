import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Function to train and evaluate a classifier
def train_evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Class to encapsulate the classifier chain
class ClassifierChain:
    def __init__(self, base_clf, num_chains=5):
        self.base_clf = base_clf
        self.num_chains = num_chains
        self.chains = [make_pipeline(StandardScaler(), base_clf()) for _ in range(num_chains)]

    def fit(self, X, y):
        for chain in self.chains:
            chain.fit(X, y)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.num_chains))
        for i, chain in enumerate(self.chains):
            predictions[:, i] = chain.predict(X)
        return predictions.mean(axis=1)
    
def main():
    # Initialize variables
    bay_clf_chain_clf_dict = {}
    num_runs = 10
    X, y = np.random.rand(100, 10), np.random.randint(0, 2, 100)  # Example data

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define classifiers
    classifiers = {
        'RandomForest': RandomForestClassifier,
        'LogisticRegression': LogisticRegression,
        'GaussianNB': GaussianNB
    }

    # Run the experiments
    for run in range(num_runs):
        for clf_name, clf_class in classifiers.items():
            clf_chain = ClassifierChain(base_clf=clf_class)
            clf_chain.fit(X_train, y_train)
            y_pred = clf_chain.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred > 0.5)
            if clf_name not in bay_clf_chain_clf_dict:
                bay_clf_chain_clf_dict[clf_name] = []
            bay_clf_chain_clf_dict[clf_name].append(accuracy)

    # Print results
    for clf_name, accuracies in bay_clf_chain_clf_dict.items():
        print(f"{clf_name}: Mean Accuracy = {np.mean(accuracies)}, Std = {np.std(accuracies)}")

if __name__ == "__main__":
    main()