# Music Classification Implemented in Pytorch
This repository is a part of music genre classification.
Companies nowadays use music classification, either to be able to place recommendations to their customers 
(such as Spotify, Soundcloud) or simply as a product (for example Shazam).
Determining music genres is the first step in that direction.\
Machine Learning techniques have proved to be quite successful in extracting trends and patterns from the large pool of data. 
The same principles are applied in Music Analysis also.

The data was obtained from:
[FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma)

The model was inspired from:
[Convolutional Recurrent Neural Networks for Music Classification](https://arxiv.org/abs/1609.04243)

We used both *CNN* and *RNN* in order to classify the music genres from the spectrograms of the audio files.

#### **Note**
The code was tested on *Ubuntu LTS 18.04* and *Python 3.6.9*  
## Installation and Running
First clone the repository and enter it.
```
git clone https://github.com/DavidSriker/MusicClassification_Pytorch
cd MusicClassification_Pytorch
```
Before downloading the data, `7z` need to be installed.
```
sudo apt-get update
sudo apt-get install p7zip-full
```
#### Data
Once installed run the `DataCollect.sh` script.
```
./DataCollect.sh
```
This will create a *Data* directory inside the *MusicClassification_Pytorch* directory.
This will download the *small* data base, but with a few changes its possible to run with the *medium* or *large*
#### Training
First install all the requiremnts.
```
pip install --upgrade pip
pip install -r pip_require.txt
```
Once done one can run the following: `python3 Train.py -h` to see the possible flags.
To run the training process:
```
python3 Train.py --epochs 100 --batch_size 5
```
#### Testing
Only once the training has finished, a model is saved in a create directory named *TrainedModel*.\
inside the directory there is 2 files:
* Model parameters named: ''MusicClassifer_E_<Number of Epochs>_BS_<Batch Size>.pt''
* Trainig history named:  ''MusicClassifer_E_<Number of Epochs>_BS_<Batch Size>_history.txt''

To run the testing script run it with flags that matches the saved model; to see the optional flags run `python3 Test.py -h`.\
To run the testing process:
```
python3 Test.py --epochs 100 --batch_size 5
```
Then a confusion matrix will be added to the main directory under the name ''confusionMatrix.png''

## Results
The model was trained for **100** epochs with batch size of **5**.\
The Training process did not pass the 50% accuracy.\
The training/validation loss can be seen in the following plot obtained from the history file.

![Image description](lossGraph.png?raw=true)

The training/validation accuracy can be seen in the following plot obtained from the history file.

![Image description](accuracyGraph.png?raw=true)

The test confusion matrix is:

![Image description](confusionMatrix.png?raw=true)


## 
-[x]  Pre-Process the raw data
-[x]  Implement CRNN model
-[ ]  Implemnt a parallel CRNN model, [Acoustic Scene Classification 
Using Parallel Combination of LSTM and CNN](https://pdfs.semanticscholar.org/4e7d/ad845bd9e1d399bf729724442cb7404549d1.pdf)
-[x]  Implement Train script
-[x]  Implement Test script
-[x]  Export plots and confusion matrix
-[ ]

---
### Author
* David Sriker - *David.Sriker@gmail.com*
