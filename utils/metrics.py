

def all_class_report(logger, preds, targets, cls_num, cls_names):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    logger.log(f"{'Class':<15}{'TP':<8}{'FP':<8}{'P':<8}{'R':<8}{'F1':<8}{'Support'}")
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(cls_num))
    )
    cm = confusion_matrix(targets, preds, labels=list(range(cls_num)))
    for i, name in enumerate(cls_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        logger.log(f"{name:<15}{tp:<8}{fp:<8}{precision[i]:<8.3f}{recall[i]:<8.3f}{f1[i]:<8.3f}{support[i]}")
    return cm
