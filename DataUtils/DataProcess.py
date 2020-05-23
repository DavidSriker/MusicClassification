import numpy as np
from torch.utils import data
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score
from DataUtils.DataPreProcess import *


def npzFileLoader(path):
    npz_file = np.load(path)
    return npz_file['arr_0'], npz_file['arr_1']


def reverseLabelDict(labelDict):
    return {v: k for k, v in labelDict.items()}


def oneHotVector(gt):
    n_values = np.max(gt) + 1
    return np.eye(n_values)[gt]


def createDataLoader(dataArr, gtArr, batchSize=2, train=True, workers=0):
    data_tensor = torch.Tensor(dataArr.reshape((dataArr.shape[0], 1, *dataArr.shape[1:])))  # transform to torch tensor
    gt_tensor = torch.Tensor(gtArr)

    dataset = data.TensorDataset(data_tensor, gt_tensor)
    return data.DataLoader(dataset, batch_size=batchSize, shuffle=train, num_workers=workers)
    # return dataset

def plotConfusionMatrix(gt, pred):
    plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    confusion_mat = confusion_matrix(gt, pred)
    print(confusion_mat)
    sns.heatmap(confusion_mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=GENRES_CODES.keys(),
                yticklabels=GENRES_CODES.keys())
    plt.yticks(rotation=45)
    plt.xticks(rotation=45)
    plt.xlabel('True Label', fontweight='bold')
    plt.ylabel('Predicted Label', fontweight='bold')
    plt.title("Confusion Matrix", fontweight='bold')
    plt.tight_layout()
    plt.savefig("confusionMatrix.png")
    return

def plotHistory(history):
    plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(history["Epochs"], history["Train Loss"], '-g')
    plt.plot(history["Epochs"], history["Validation Loss"], '-b')
    plt.title('Model Loss', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig("lossGraph.png")

    plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(history["Epochs"], history["Train Acc"], '-g')
    plt.plot(history["Epochs"], history["Validation Acc"], '-b')
    plt.title('Model Accuracy', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig("accuracyGraph.png")
    return
