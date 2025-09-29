from src.utils import load_dataset, standardize_fit, standardize_apply
from src.svm_qp import SVM_QP

def evaluate(name, clf, X, y):
    acc = (clf.predict(X) == y).mean()
    print(f"{name} accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    #load full train.csv and test.csv
    X_all, y_all = load_dataset("data/train.csv")
    X_test, y_test = load_dataset("data/test.csv")

    #data split: first 4000 -> train, remaining -> val
    X_train, y_train = X_all[:4000], y_all[:4000]
    X_val,   y_val   = X_all[4000:], y_all[4000:]

    #standardize using TRAIN stats only
    X_train_std, mu, sigma = standardize_fit(X_train)
    X_val_std  = standardize_apply(X_val,  mu, sigma)
    X_test_std = standardize_apply(X_test, mu, sigma)

    #best hyperparameters from validation search: Linear, C = 0.01
    clf = SVM_QP(C=0.01, kernel=("linear", {}))

    print("[FINAL] Fitting dual linear SVM (C=0.01) on the 4,000-sample training set...")
    clf.fit(X_train_std, y_train)

    #report Train / Val / Test (Test is the official final)
    evaluate("Train", clf, X_train_std, y_train)
    evaluate("Val",   clf, X_val_std,   y_val)
    evaluate("Test (FINAL)", clf, X_test_std, y_test)
