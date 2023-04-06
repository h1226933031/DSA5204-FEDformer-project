import numpy as np
import matplotlib.pyplot as plt


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    return mae, mse


def visual(true, preds=None, true_label='GroundTruth', preds_label='Prediction', name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label=true_label, linewidth=2)
    if preds is not None:
        plt.plot(preds, label=preds_label, linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
