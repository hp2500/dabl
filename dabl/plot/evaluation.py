import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix

from .utils import heatmap


def _get_decision_values(est, X):
    if len(est.classes_) != 2:
        raise ValueError("Only binary classification supported for now")
    if hasattr(est, 'decision_function'):
        return est.decision_function(X)
    return est.predict_proba(X)[:, 1]


def plot_precision_recall(est, X_val, y_val, ax=None):
    """Computes and plots the precision-recall curve.
    """
    dec = _get_decision_values(est, X_val)
    precision, recall, thresholds = precision_recall_curve(y_val, dec)
    if ax is None:
        ax = plt.gca()
    ax.plot(recall, precision)
    ax.set_ylabel('precision')
    ax.set_xlabel('recall')
    ax.set_title('Precision-Recall Curve')


def plot_confusion_matrix(est, X_val, y_val, ax=None):
    y_pred = est.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    heatmap(cm, ax=ax, xlabel='True label', ylabel='predicted label',
            xticklabels=est.classes_, yticklabels=est.classes_, origin='upper')
