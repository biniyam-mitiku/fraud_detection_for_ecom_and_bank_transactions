import numpy as np

def get_shap_matrix(explainer, X):
    """Return (shap_values[n, f] in log-odds for the fraud class, base_value)."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # older versions: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # some versions: (n, f, 2)
        sv = sv[:, :, 1]
    base = np.ravel(explainer.expected_value)[-1]
    return sv, float(base)

def check_additivity(model, X, sv, base, tol=1e-3):
    raw = model.predict(X, raw_score=True)
    gap = np.abs(raw - (base + sv.sum(axis=1))).max()
    print(f"max |raw score - (base + sum of SHAP)| = {gap:.2e}")
    assert gap < tol, "additivity broken: wrong model, wrong column order, or wrong output shape"
    return gap
