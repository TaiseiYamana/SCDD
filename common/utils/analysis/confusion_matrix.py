import torch
import matplotlib
import pandas as pd
import seaborn as sns
import torch.nn.functional as F

matplotlib.use('Agg')

class Cal_ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y = []
        self.t = []
    def update(self, y, t):
        _, y =  torch.max(F.softmax(y, dim =1), dim = 1)
        y = y.to('cpu').detach().numpy().copy().astype('int64')
        y = y.tolist()
        self.y.extend(y)


        t = t.to('cpu').detach().numpy().copy().astype('int64')
        t = t.tolist()
        self.t.extend(t)

def plot_cm(cm, normalize=False, title='Confusion matrix', annot=False, cmap='YlGnBu'):
    import pandas as pd
    import seaborn as sns

    data = cm.matrix
    if normalize:
        title += '(Normalized)'
        data = cm.normalized_matrix

    df = pd.DataFrame(data).T.fillna(0)
    ax = sns.heatmap(df, annot=annot, cmap=cmap, fmt='d')
    ax.set_title(title)
    ax.set(xlabel='Predict', ylabel='Actual')
