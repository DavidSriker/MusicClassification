import argparse
from Models.SerialCRNN import *
from DataUtils.DataPreProcess import *
from DataUtils.DataProcess import *
import torch.optim as optim
import json
import copy as cp


def updateHistory(history, epoch, trainAcc, validAcc):
    history["Epochs"].append(epoch)
    history["Train Loss"].append(1 - trainAcc)
    history["Validation Loss"].append(1 - validAcc)
    history["Train Acc"].append(trainAcc)
    history["Validation Acc"].append(validAcc)
    return


def saveModel(model, history, name, path="TrainedModel"):
    current_path = os.getcwd()
    save_dir = os.path.join(current_path, path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    torch.save(model.state_dict(), os.path.join(save_dir, name + ".pt"))
    json.dump(history, open(os.path.join(save_dir, name + "_history.json"), "w"))
    return

def trainSerial(trainData, trainGT, validationData, validationGT, numEpochs=100, batchSize = 2):

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_cuda:
        trainLoader = createDataLoader(trainData, trainGT, batchSize=batchSize, workers=2)
        validationLoader = createDataLoader(validationData, validationGT, batchSize=batchSize, workers=2, train=False)
    else:
        trainLoader = createDataLoader(trainData, trainGT, batchSize=batchSize)
        validationLoader = createDataLoader(validationData, validationGT, batchSize=batchSize, train=False)

    model = SerialCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Loop over epochs
    history ={"Epochs":[],
              "Train Loss": [],
              "Validation Loss": [],
              "Train Acc": [],
              "Validation Acc": []}
    valid_acc_max = 0.0
    print_freq = 500
    best_model = cp.deepcopy(model)
    for epoch in range(numEpochs):
        print(10 * "=" + " Epoch Number [{:}] ".format(epoch + 1) + 10 * "=")
        print("Learning rate is: {:.13f}".format(scheduler.get_lr()[0]))

        print("<~~~~   Epoch {:} / {:} Results   ~~~~>".format(epoch + 1, numEpochs))
        validation_accuracy = validateSerial(validationLoader, model, device)
        train_accuracy = validateSerial(trainLoader, model, device)
        print("Train Loss: {:.3f}, Train Accuracy: {:.3f}".format(1 - train_accuracy, train_accuracy))
        print("Validation Loss: {:.3f}, Validation accuracy: {:.3f}".format(1 - validation_accuracy, validation_accuracy))
        updateHistory(history, epoch, train_accuracy, validation_accuracy)
        if valid_acc_max <= validation_accuracy:
            best_model = cp.deepcopy(model)

        # Training
        running_loss = 0.0
        for idx, (data_batch, labels_batch) in enumerate(trainLoader):
            # Transfer to GPU if possible
            data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            predictions = model(data_batch)
            loss = criterion(predictions, labels_batch.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if idx % print_freq == 0 and idx != 0:  # print every 500 mini-batches
                print("Batch number: {:} ----> Train Batch Loss: {:.3f}".format(idx, running_loss / (labels_batch.size(0) * print_freq)))
                running_loss = 0.0
        scheduler.step()

    print('Finished Training')
    saveModel(best_model, history, "MusicClassifer_E_{:}_BS_{:}".format(numEpochs, batchSize))
    return


def validateSerial(loader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device).long()).sum().item()

    return correct / total


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Music Genre Classifier')
    parser.add_argument('--data_path', default="Data/", type=str, help='data path')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=15, type=int, help='batch size')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(os.getcwd(), "Data", "processed_data")):
        prepareData(args.data_path)
    processed_data_path = os.path.join(os.getcwd(), "Data", "processed_data")
    train_data, train_gt = npzFileLoader(os.path.join(processed_data_path, "train_data.npz"))
    validation_data, validation_gt = npzFileLoader(os.path.join(processed_data_path, "validation_data.npz"))

    trainSerial(train_data, train_gt, validation_data, validation_gt, args.epochs, args.batch_size)

    return

if __name__ == '__main__':
    main()
