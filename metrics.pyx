# metrics.pyx

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from cython cimport boundscheck, wraparound

from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  matthews_corrcoef,
  roc_auc_score,
  confusion_matrix,
  classification_report,
)

@boundscheck(False)
@wraparound(False)
def compute_metrics_cython(np.ndarray[np.int_t, ndim=1] preds, np.ndarray[np.int_t, ndim=1] labels):
    cdef dict metrics = {}
    cdef float acc = accuracy_score(labels, preds)
    cdef float precision = precision_score(labels, preds, average="weighted", zero_division=0)
    cdef float recall = recall_score(labels, preds, average="weighted", zero_division=0)
    cdef float f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    cdef float mcc = matthews_corrcoef(labels, preds)
    try:
        cdef float roc_auc = roc_auc_score(labels, preds, average="weighted", multi_class="ovr")
    except:
        cdef float roc_auc = float("nan")
    cdef np.ndarray[np.int_t, ndim=2] cm = confusion_matrix(labels, preds)
    cdef dict report = classification_report(labels, preds, zero_division=0, output_dict=True)
    metrics["accuracy"] = acc
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1
    metrics["mcc"] = mcc
    metrics["roc_auc"] = roc_auc
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = report
    return metrics
