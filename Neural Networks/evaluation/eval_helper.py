# import standard libraries
import numpy as np
import matplotlib.pyplot as plt

# import evaluation libraries
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# define functions for plotting confusion matrix statistics
def plot_classification_metrics(labels, predictions):

    # define and plot confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["not popular", "popular"])
    display.plot()
    plt.show()

    # plot ROC and calculate AUC
    fpr, tpr, _ = metrics.roc_curve(labels, predictions)
    auc = metrics.roc_auc_score(labels, predictions)
    plt.plot(fpr, tpr, label="AUC: "+str(auc))
    plt.xlabel("False Postitive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc = "best")
    plt.show()


# define function for evaluating models
def evaluate_model(loader, model, loss_func, device):
    predictions = []
    labels = []
    losses = []
    for index, batch in enumerate(loader):
        embedding, score = batch

        # make sure data is on device
        embedding = embedding.to(device)
        scores = score.to(device)

        # get prediction
        preds = model(embedding)

        # get loss
        loss = loss_func(preds, scores.float().unsqueeze(1))

        # add data to outputs
        predictions.append(preds)
        labels.append(scores)
        losses.append(loss)
    
    return (predictions, labels, losses)



# define functions for plotting losses
def plot_losses(losses, title):
    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Iteration")
    plt.ylabel(title)
    plt.show()

def plot_losses_log(losses, title):
    plt.figure(figsize=(20, 10))
    plt.plot(np.log(np.arange(len(losses))), losses)
    plt.xlabel("Log Iteration")
    plt.ylabel(title)
    plt.show()