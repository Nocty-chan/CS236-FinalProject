# CS236- Final Project repository 

Final Report: https://drive.google.com/open?id=18YWWARGSbJyWrmbSm1JmQH51lEIJoleU

## Training AC-GAN:
> python train_acgan.py

## Training classifier:
> python main.py 

## Training classifier with FGSM 
> python main.py --adversarial_mode fgsm 

## Training classifier with GAN samples 
> python main.py --adversarial_mode gan --gen_name <name of GAN to use>

## Training classifier with unrestricted adversarial examples 
> python train_unrestricted.py
