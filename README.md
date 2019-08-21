# Adversarial training with Unrestricted Adversarial
## CS236 Final Project
Final Report: https://drive.google.com/open?id=18YWWARGSbJyWrmbSm1JmQH51lEIJoleU

## Motivation 

Dataset augmentation is a well-known technique used to increase the generalization power of machine learning models by introducing samples derived from the original training set. 
However, it is not obvious which augmentation techniques can be helpful: while simple geometric transformations like translations, mirroring help increase the accuracy of image classifiers, introducing adversarial examples at train time is detrimental to the accuracy on clean inputs. 

This work focuses on leveraging generative models to perform data augmentation. We study the case of image classification on the CIFAR-100 dataset and compare different augmentation methods using deep generative models. In particular, we study the effect of augmenting the training set with unrestricted adversarial samples.

## Task definition

Our goal is to compare different augmentation techniques in the adversarial training setting where the loss is given as:
%TODO: insert loss 

We train on the CIFAR-100 dataset, a set of 60,000 32x32 color images labeled with 100 classes. Our baseline model is a DenseNet trained with the usual cross-entropy loss.  

## Augmentation techniques 
### Generative model
For our generative model, we choose the Auxiliary Classifier GAN trained with Gradient Penalty. The generator and discriminator use ResNet layers. The resulting model allows us to generate samples from a chosen class that we can use for data augmentation 

### Adversarial Samples
Adversarial samples are typically obtained by adding an imperceptible adversarial noise to an input image, which makes it difficult for the classifier to correctly classify. An easy way to defend against adversarial samples is to augment the training set with adversarial samples, but this has been shown to decrease the accuracy of the model on clean samples. Here, for comparison, we test data augmentation with adversarial samples generated with FGSM (Fast Gradient Sign Method)

### Unrestricted Adversarial Samples
We generate unrestricted adversarial samples using our trained generative model by optimizing the latent variable z to produce an image that will be misclassified by the classifier. Details can be found in the final report. 



## Results


## Repository Usage 
### Pre-requisites

### Training AC-GAN:
> python train_acgan.py

### Training classifier:
> python main.py 

## #Training classifier with FGSM 
> python main.py --adversarial_mode fgsm 

### Training classifier with GAN samples 
> python main.py --adversarial_mode gan --gen_name <name of GAN to use>

### Training classifier with unrestricted adversarial examples 
> python train_unrestricted.py
