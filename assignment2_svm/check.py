import numpy as np
from src.utils import load_dataset, standardize_fit, standardize_apply
from src.svm_qp import SVM_QP

# 1) Load a tiny slice to keep it fast
X_train_full, y_train_full = load_dataset("data/train.csv")
X = X_train_full[:300]      # 300 samples for a smoke test
y = y_train_full[:300]

# 2) Train/val split (here: 200 train, 100 val)
X_tr, y_tr = X[:200], y[:200]
X_va, y_va = X[200:], y[200:]

# 3) Standardize using TRAIN stats only
X_tr_std, mu, sigma = standardize_fit(X_tr)
X_va_std = standardize_apply(X_va, mu, sigma)

# 4) Fit a simple linear SVM (no tuning yet)
clf = SVM_QP(C=1.0, kernel=("linear", {}))
clf.fit(X_tr_std, y_tr)

# 5) Evaluate on train and val
yhat_tr = clf.predict(X_tr_std)
yhat_va = clf.predict(X_va_std)

acc_tr = (yhat_tr == y_tr).mean()
acc_va = (yhat_va == y_va).mean()

print(f"Train acc: {acc_tr:.3f}")
print(f"Val   acc: {acc_va:.3f}")
