# DCGAN for Generating Music

This is a deep convolutional GAN adapted to work on our image representations of music. Some sources I used for the code / reading about DCGANs:
* https://github.com/carpedm20/DCGAN-tensorflow
* https://towardsdatascience.com/implementing-a-generative-adversarial-network-gan-dcgan-to-draw-human-faces-8291616904a

## Requirements
Tested with Python 3.6.5. Install dependencies with
```
pip install -r requirements.txt
```

## Training
Run with 
```
python main.py
```
Currently, the script expects for input image files to be 24 x 24 and in a folder called `midi24`. My current dataset can be found [here](https://www.dropbox.com/s/9f5z46aln8kam5c/midi24.zip?dl=0). 

It outputs samples to `samples/`. 

## To do:
* Mode collapse??
* Figure out hyperparams