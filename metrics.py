import math
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
#-----------------------------------------------------------------------------------------
import subprocess


def mse(y1, y2):
    return ((y1 - y2) ** 2).mean()
#-----------------------------------------------------------------------------------------
def rmse(y1, y2):
    return math.sqrt(mse(y1, y2))
#-----------------------------------------------------------------------------------------
def rsq(y_pred, y_actual):
    u = ((y_pred - y_actual) ** 2).mean()
    v = ((y_actual.mean() - y_actual) ** 2).mean()
    return 1 - u / (v + 1e-9)
#-----------------------------------------------------------------------------------------
def print_scores(rf, _X_train, _X_valid, _y_train, _y_valid):
    print([
        rmse(rf.predict(_X_train), _y_train),
        rmse(rf.predict(_X_valid), _y_valid),
        rf.score(_X_train, _y_train),
        rf.score(_X_valid, _y_valid),
        rf.oob_score_ if hasattr(rf, "oob_score_") else ''
    ])
    return rf
#-----------------------------------------------------------------------------------------
def visualize_tree(dt, figsize=(20, 20), feature_names=None):
    export_graphviz(dt,
                   out_file="iris_tree.dot",
                   feature_names=feature_names,
                   rounded=True,
                   filled=True)
    subprocess.call(['dot', '-Tpng', 'iris_tree.dot', '-o', 'iris_tree.png'])

    plt.figure(figsize = figsize)
    plt.imshow(plt.imread('iris_tree.png'))

