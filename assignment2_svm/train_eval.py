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

def grid_search_rbf(X_train, y_train, X_val, y_val, C_values, gamma_values):
    
    #tune RBF kernel SVM over C and gamma using the validation set.
    #returns (best_params, results_list) where best_params = {"C": ..., "gamma": ...}
    
    from src.svm_qp import SVM_QP
    results = []
    for C in C_values:
        for gamma in gamma_values:
            print(f"[TUNE] RBF SVM with C={C}, gamma={gamma} ...")
            clf = SVM_QP(C=C, kernel=("rbf", {"gamma": gamma}))
            clf.fit(X_train, y_train)
            val_acc = (clf.predict(X_val) == y_val).mean()
            results.append({"kernel": "rbf", "C": C, "gamma": gamma, "val_acc": float(val_acc)})

    #sort by validation accuracy (desc)
    results.sort(key=lambda r: r["val_acc"], reverse=True)
    best = results[0]
    print("\nValidation results (RBF kernel):")
    for r in results:
        print(f"  C={r['C']:>6}, gamma={r['gamma']:<8} | val_acc={r['val_acc']:.4f}")
    print(f"\n[BEST-RBF] C={best['C']}, gamma={best['gamma']} with val_acc={best['val_acc']:.4f}")
    return {"C": best["C"], "gamma": best["gamma"]}, results


if __name__ == "__main__":
    # Load standardized splits per assignment spec
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split()

    # --- RBF kernel grid only ---
    from src.svm_qp import SVM_QP

    # Small, sensible grid first (expand later only if it wins)
    C_grid_rbf = [0.01, 0.1, 1.0]
    gamma_grid = [0.001, 0.01, 0.1]

    best_rbf_params, rbf_results = grid_search_rbf(
        X_train, y_train, X_val, y_val,
        C_grid_rbf, gamma_grid
    )

    print("\n==> Training best RBF model for reference ...")
    best_rbf = SVM_QP(C=best_rbf_params["C"], kernel=("rbf", {"gamma": best_rbf_params["gamma"]}))
    best_rbf.fit(X_train, y_train)
    print("RBF model performance:")
    evaluate_model(best_rbf, X_train, y_train, "Train")
    evaluate_model(best_rbf, X_val, y_val, "Val")
    evaluate_model(best_rbf, X_test, y_test, "Test")
