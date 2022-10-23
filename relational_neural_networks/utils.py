import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np


def plot_confusion_matrices(y_test, test_pred):
    '''plot confusion matrices with each choice of normalization'''

    fig, axs = plt.subplots(figsize=(18,3.5), ncols=4)

    for ax, normalize in zip(axs, ('true', 'pred', 'all', None)):

        cm = sklearn.metrics.confusion_matrix(y_test, test_pred, normalize=normalize)
        sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=('non-set', 'set')).plot(ax=ax);
        ax.set_title(f'Normalization: {normalize}');

    for ax in axs[1:]:
        ax.set_ylabel('')

    return fig


def plot_history(history, plot_attrs, val=True, **plt_kwargs):
    '''plot given attributes from training history'''

    fig, axs = plt.subplots(ncols=len(plot_attrs), **plt_kwargs)

    if not all(plot_attr in history.history for plot_attr in plot_attrs):
        raise ValueError('not all `plot_attrs` are in the history object')

    for plot_attr, ax in zip(plot_attrs, axs):
        ax.plot(history.history[plot_attr], label=plot_attr)
        if val:
            ax.plot(history.history[f'val_{plot_attr}'], label=f'val_{plot_attr}')
        ax.set_ylabel(plot_attr)
        ax.set_xlabel('epoch')
        ax.legend(loc='upper right')

    return fig

def plot_roc_pr_curves(pred_probas, y, **kwargs):
    '''create subplots fig for ROC and PR curves'''

    pred = pred_probas[:, 1]

    subplot_kwargs = {'figsize': (8,3)}
    subplot_kwargs.update(kwargs)
    fig, (ax1, ax2) = plt.subplots(ncols=2, **subplot_kwargs)
    sklearn.metrics.RocCurveDisplay.from_predictions(y, pred, ax=ax1)
    sklearn.metrics.PrecisionRecallDisplay.from_predictions(y, pred, ax=ax2)
    return fig

def print_classification_report(model, X, y, **kwargs):
    pred = np.argmax(model(X), axis=1)
    print(sklearn.metrics.classification_report(y, pred, **kwargs))
