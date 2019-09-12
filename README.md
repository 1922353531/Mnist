# Mnist
## Abstract
This is a primary projects of DeepLearning written by Python3, which can help you get start in DeepLearning. This projects is implemented by Tensorflow and CNN(Convlution neural network) and RNN(Recurrent Neural Network). And you can learn them here.
## How to get start
### 1.Structure
1. MNIST_data contains MNIST'datasets and if you run the model first it will be downloaded automactically.
2. Mnist_parameter_CNN contains retrained parameters of CNN model and you can use them directly.
3. Mnist_parameter_RNN contains retrained parameters of RNN model and you can use them directly.
4. CNN_Trained_Model_1.0.py allows you use the retrained CNN's parameters to predict in the Mnist_datasets.
5. CNN_Trained_Model_2.0.py allows you use the retrained CNN's parameters to predict in the Mnist_datasets and your own pictures.
6. CNN_Training_Model.py uses Mnist'datasets to train neural based on CNN model. You can run it to get CNN's new parameters and you can also change its structure.
7. RNN_Training_Model.py uses Mnist'datasets to train neural based on RNN model. You can run it to get RNN's new parameters and you can also change its structure.
### 2.Training in your own
If you want to use CNN model, you can run CNN_Training_Model.py directly and the new trained parameters will be saved in Mnist_parameter_CNN. And if you want to use RNN model, just run RNN_Training_Model is ok.
### 3.Using retrained network
There are only CNN retrained model in this project and you can run CNN_Trained_Model_1.0.py or CNN_Trained_Model.py directly. If you want to use retrained RNN model, you can train it by yourself.
## PS
Due to the training datasets is pure Mnist, maybe you will find the retrained network can't recognition some wordarts. This problem can be sovled by using larger and more variable datasets.
