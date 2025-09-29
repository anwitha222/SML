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

def grid_search_linear_C(X_train, y_train, X_val, y_val, C_values):
    
    #hyperparameter tuning
    #tune only C for the linear kernel using the validation set
    #returns (best_C, results_list) where results_list is a list of dicts
    
    from src.svm_qp import SVM_QP
    results = []

    for C in C_values:
        print(f"[TUNE] Training linear SVM with C={C} ...")
        clf = SVM_QP(C=C, kernel=("linear", {}))
        clf.fit(X_train, y_train)
        val_acc = (clf.predict(X_val) == y_val).mean()
        results.append({"kernel": "linear", "C": C, "val_acc": float(val_acc)})

    #sort by validation accuracy (desc)
    results.sort(key=lambda r: r["val_acc"], reverse=True)
    best = results[0]
    best_C = best["C"]
    print("\nValidation results (linear kernel):")
    for r in results:
        print(f"  C={r['C']:>6}  |  val_acc={r['val_acc']:.4f}")
    print(f"\n[BEST] C={best_C} with val_acc={best['val_acc']:.4f}")
    return best_C, results

if __name__ == "__main__":
    #load standardized splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split()

    #grid search: linear kernel over a small C grid 
    C_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_C, results = grid_search_linear_C(X_train, y_train, X_val, y_val, C_grid)

    #train final linear model with best C on TRAIN only (for visibility)
    final_clf = SVM_QP(C=best_C, kernel=("linear", {}))
    final_clf.fit(X_train, y_train)

    #report train/val/test (test is just for a quick look here... the final workflow will retrain on train+val)
    print("\nFinal evaluation with best C (TRAIN-only fit):")
    evaluate_model(final_clf, X_train, y_train, "Train")
    evaluate_model(final_clf, X_val, y_val, "Val")
    evaluate_model(final_clf, X_test, y_test, "Test")

