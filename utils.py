from sklearn.metrics import auc, roc_curve, accuracy_score
import matplotlib.pyplot as plt

def plot_roc(y_true, y_pred, model_name=""):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic '+model_name)
    plt.legend(loc="lower right")
    return plt.show();

def get_metrics(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr,tpr)
    metrics = {}
    metrics["auc"] = roc_auc
    y_bin = (y_pred>0.5).astype(int)
    acc = accuracy_score(y_true, y_bin)
    metrics["accuracy (lim=0.5)"] = acc
    return metrics
    