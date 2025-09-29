from src.utils import load_dataset, standardize_fit, standardize_apply
from src.svm_qp import SVM_QP
import numpy as np

def load_and_split():
    #load train abd test data
    X_all, y_all = load_dataset("data/train.csv")
    X_test, y_test = load_dataset("data/test.csv")

    #split: first 4000 for training and the rest for validation
    X_train, y_train = X_all[:4000], y_all[:4000]
    X_val, y_val = X_all[4000:], y_all[4000:]

    #standardize using train stats only
    X_train_std, mu, sigma = standardize_fit(X_train)
    X_val_std = standardize_apply(X_val, mu, sigma)
    X_test_std = standardize_apply(X_test, mu, sigma)

    return (X_train_std, y_train,
            X_val_std, y_val,
            X_test_std, y_test)

def evaluate_model(clf, X, y, name="set"):
    yhat = clf.predict(X)
    acc = (yhat == y).mean()
    print(f"{name} accuracy: {acc:.3f}")
    return acc

if __name__ == "__main__":
    #load splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split()

    #try a simple linear kernel first
    clf = SVM_QP(C=1.0, kernel=("linear", {}))
    clf.fit(X_train, y_train)

    #evaluate
    evaluate_model(clf, X_train, y_train, "Train")
    evaluate_model(clf, X_val, y_val, "Val")
    evaluate_model(clf, X_test, y_test, "Test")
