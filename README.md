# WAE
An implementation of Wasserstein Autoencoder. In this work, I have focused on the WAE-GAN variant. In this implementation the encoder is implemented as a dirac measure. However, the paper theoretically claims that their approach can be extended to probabilistic encoders as well. 

# Model Weights 
Model weights can be downloaded from [here](https://drive.google.com/drive/folders/1l_SY9c_50km9tgqGzub8lyYMadF6Z7HX?usp=sharing).
In the given^ link you will find weights for each of the model trained on celebA, MNIST and CIFAR10 dataset.

## Setup
* Python 3.5+
* Tensorflow 1.9

## Relevant Code Files

File config.py contains the hyper-parameters for WAE-GAN reported results.

File wae_gan.py contains the code to both train and test WAE-GAN model. For training call train function.

## Usage
### Training a model
NOTE: For celebA, make sure you have the downloaded dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and keep it in the current directory of project.
```
python wae_gan.py
```

### Test a trained model 
Just comment the train() function call in 
First place the model weights in model_directory (mentioned in vae-gan_inference.py) and then:
```
python wae-gan.py 
```
## Generations

MNIST            |  Celeb-A 
:-------------------------:|:-------------------------:|
![](https://prateekmunjal.github.io/img/wae/generations_mnist.gif)  |  ![](https://prateekmunjal.github.io/img/wae/generations_celeba.gif)
