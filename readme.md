# DFO Bot (Big Otolith Tensor?)
Deep convolutional network for rapid automated otolith aging

## Directory structure:

Two main directories `preprocessing` and `model`.  Preprocessing is used to standardize and organize input images into a common format for the network, namely single otoliths cropped and then resized into a fixed resolution (e.g. 1000x1000).

Model contains the training and validation workflows for the ML model, as well as various helpers for running the scripts and performing the hyperparameter search.  

## Installation:
On Ubuntu need to install latest nvidia drivers (or whatever is suitable for the graphics card): `sudo apt install nvidia-driver-???` and a `pip install -r requirements.txt`


## References:
All the machine learning knowledge required is are covered in lectures 1-12 of this course:
https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/

This methods used in this code are not anything novel, they closely follow the aproach employed by Moen et al. 2018 and other recent otolith aging work:

Moen, E., et al. (2018). "Automatic interpretation of otoliths using deep learning." PLoS One 13(12): e0204713.

Sigurðardóttir, A. R., et al. (2023). "Otolith age determination with a simple computer vision based few-shot learning method." Ecological Informatics 76.

Politikos, D. V., et al. (2021). "Automating fish age estimation combining otolith images and deep learning: The role of multitask learning." Fisheries Research 242.

## Model:
The current model structure consists of a pretrained version of ResNet50 with a single output instead of 1000 classes and uses an MSE loss.
Current data sets include American Plaice otoliths from the 2023 RV survey (~3500) and Herring otoliths from the 2019 season (4500).
Images containing an otolith pair are split into two separate images resulting in a combined total training set of ~15000 images.
Basic image augmentation techniques (random rotation, cropping) and image normalization are implemented through PyTorch's dataloader class.  
The model trains on otoliths from both species simultaneously to maximize the overall generality of the model, best results so far have been in the ~65% accuracy range.  


# October 2025:
Switch to using yellowtail dataset, ~10000 images across 20 years of RV survey results.


# Creating a new CNN:
 - Preprocess images into single raw dir (crop_and_isolate).  Each image should have a UUID filename
 - Create csv of labels, one row per UUID.
 - Create a train/test folder (train_test_splitter)

## Classifiers
Will create several 'helper' CNN's to improve data quality and provide potentially superior initial weights for the aging CNN.  
The first of these helper models is the "crack finder".  This required ~2000 labeled images of individual otoliths with one of four labels: good, cracked, twinned (images contain both otoliths) and crystal (otoliths with a bubbly texture).
Note that there was a fairly low hit count for all of the classes other than good (135 cracked, 35 crystal, 10 twin'd).  A hyper parameter search was performed, optimizing for the highest accuracy at detecting cracks. The optimal parameters were:
 - Learning rate: 1e-4
 - Weight decay 1e-6
 - Image crop size: 300x300
15 epochs was sufficient to obtain a 100% accuracy at crack detection which held up running on the full 20,000 image dataset

 
## Tuning:
 - Note that resnets only work on one image size once trained. i.e. if 400x400 images are used to train the model, tbe outputs will only be valid on other 400x400 images
 - The imageK weights are an important starting point, the model trains far worse without transfer learning
 - 