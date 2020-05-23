import argparse
from Models.SerialCRNN import *
from DataUtils.DataPreProcess import *
from DataUtils.DataProcess import *
import json


def testSerial(testData, testGT, numEpochs=100, batchSize = 2):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if use_cuda:
        testLoader = createDataLoader(testData, testGT, batchSize=batchSize, workers=5)
    else:
        testLoader = createDataLoader(testData, testGT, batchSize=batchSize)
    net_name = "MusicClassifer_E_{:}_BS_{:}.pt".format(numEpochs, batchSize)
    model_path = os.path.join(os.getcwd(), "TrainedModel", net_name)
    if os.path.exists(model_path):
        model = SerialCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        with torch.no_grad():
            all_predictions = torch.tensor([]).to(device)
            all_gt = torch.tensor([]).to(device)
            for data_batch, labels_batch in testLoader:
                all_gt = torch.cat((all_gt, labels_batch.to(device)), dim=0)
                predictions = model(data_batch.to(device))
                all_predictions = torch.cat((all_predictions, predictions.argmax(dim=1).float()), dim=0)
        plotConfusionMatrix(all_gt.cpu().numpy(), all_predictions.cpu().numpy())
        print("Test Accuracy is: ", accuracy_score(all_gt.cpu().numpy(), all_predictions.cpu().numpy()))

    else:
        print("Cant find model in:\n{:}".format(model_path))
        exit(-1)

    return


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
    test_data, test_gt = npzFileLoader(os.path.join(processed_data_path, "test_data.npz"))

    # testSerial(test_data, test_gt, args.epochs, args.batch_size)
    # plot history
    history_file = os.path.join(os.getcwd(),
                                "TrainedModel",
                                "MusicClassifer_E_{:}_BS_{:}_history.json".format(args.epochs, args.batch_size))
    if os.path.exists(history_file):
        print("plotting history")
        history_data = json.load(open(history_file))
        plotHistory(history_data)

    return


if __name__ == '__main__':
    main()