# Image Classifier

The second project for the AI Programming with Python Nanodegree Program in Udacity.
This production has two parts. The first part is to be trained to recognize different flowers, the second parts is to predict the the flowers in given pictures.

## Command Line

### Training

python train.py data_directory

data_directory, the directory of the images that we used to train the Image Classifier

#### Options

--save_dir, the directory to save checkpoints
--arch, to choose architectures
--learning_rate, to set the learning rate
--hidden_units, to set the hidden units
--epochs, to set the epochs of training
--gpu, to use gpu for training

### Recognizing

python predict.py input checkpoint

input, the directory of the image that content the flowers to get predicted by our trained Image Classifier
checkpoint, the directory to load checkpoints

#### Options

--top_k, the number of top possibilities we want to return from the predicting
--category_names, the directory of mapping of categories to real names
--learning_rate, to set the learning rate
--gpu, to use gpu for training