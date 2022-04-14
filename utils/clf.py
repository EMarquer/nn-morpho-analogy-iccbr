def tpr_tnr_balacc_harmacc_f1(tp,tn,fp,fn):
    """Compute usefull classification statistics:
        - true positive rate (TPR), i.e., accuracy on positive samples
        - true negative rate (TNR), i.e., accuracy on negative samples
        - ballanced accuracy (balacc), i.e., mean of TPR and TNR
        - harmonic mean accuracy (harmacc), i.e., harmonic mean of TPR and TNR
        - F1-score

    :param tp: Number of true positives, i.e., number of correctly preddicted positive samples.
    :param tn: Number of true negatives, i.e., number of correctly preddicted negative samples.
    :param fp: Number of false positives, i.e., number of negative samples predicted as positive.
    :param fn: Number of false negtives, i.e., number of positive samples predicted as negtive.
    :return: TPR, TNR, balacc, harmacc, F1
    """
    tpr = tp / (tp + fn) # true positive rate i.e. accuracy on positive examples
    tnr = tn / (tn + fp) # true negative rate i.e. accuracy on negative examples
    balacc = (tpr + tnr) / 2 # mean of TPR & TNR
    harmacc = (2 * tpr * tnr) / (tpr + tnr) # harmonic mean of TPR & TNR
    f1 = (2 * tp) / (2 * tp + fp + fn) # actual F1 score

    return tpr, tnr, balacc, harmacc, f1

def mask_valid(a, b, c, d):
    eq_ = lambda x, y: (x[...,:min(x.size(-1), y.size(-1))] == y[...,:min(x.size(-1), y.size(-1))]).all(dim=-1)
    mask = (
        (eq_(a, b) & eq_(c, d)) | 
        (eq_(a, c) & eq_(b, d))
    )
    return mask