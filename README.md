# deraining

## Introduction of deraining project

#### Project mission

This project is a teamwork project proposed in the deep learning course MAP583 of  Ecole Polytechnique.

The object of this project is quite simple : raindrops in images can severely hamper the quality of vision tasks, such as semantic segmentation, it is urgent to make out the deraining project.  In this project, we apply a generative network using adversarial training to remove the raindrops in an image, and compare the semantic segmentation’s results with de-rain image and raindrop image.

![100_rain-inputs](https://github.com/ChenLyyu/deraining/blob/deraining/images/100_rain-inputs.png)![100_rain-targets](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/100_rain-targets.png)



#### Dataset presentation

The dataset can be found here :

https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K. 

The data comes from a stereo device that allows one lens to be affected by real water droplets while keeping the other lens clear. We have $861$ image pairs for training, and $239$ pairs for testing.

![rainmaker](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/rainmaker.jpg)

This dataset is provided by https://github.com/rui1996/DeRaindrop, and we compare our results with the result of article "Attentive Generative Adversarial Network for Raindrop Removal from A Single Image (CVPR 2018)[1]" in the last section.



## Network architectures 

#### Neural network representation of cGAN

This project is based on Pix2pix-tensorflow(https://github.com/affinelayer/pix2pix-tensorflow), and the neural netork architecture is cGAN.

##### Generator : U-net

The generator of cGAN is U-net, and the layer architecutre is as following :

![generator](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/generator.png)



##### Discriminator : PatchGAN

Discriminator tries to classify if each $N × N$ patch in an image is real or fake. We run this discriminator convolutionally across the image, averaging all responses to provide the ultimate output of $D$.

![discriminator](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/discriminator.png)

#### Modification based on Pix2pix

Compared to original Pix2pix, we have several modifications as following:

1. Refined the calculation graph to run test set after each epoch of training
2. Add calculations of two metrics: psnr and ssim
3. Saved the best model automatically according to two metrics
4. Add random brightness and random flip for data augementations
5. Add several kinds of attention mechanisms (p2p_channel && p2p_spatial) to the original models, however results are unsatisfactory.
6. Removed several unused functions. 

## Several Results

#### Loss function and metric

We have the loss function figure as following : 

![loss](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/loss.png)



As for the metrics , we take classic PSNR and SSIM, and we have the corresponding figure : 

![psnr](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/psnr.png)![ssim](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/ssim.png)



And the quantitative estimation of metrics is : 

![evaluation](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/evaluation.jpg)

#### Compared to article results 

Here is a direct result of our deraining model comparing with the result of article[1] : 

![compare](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/compare.jpg)

#### Segmentation representation

Then, we try to apply a pre-trained segmentation algorithm in order to evaluate qualitatively our deraining model.

![seg](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/seg.jpg)

#### Excitation

Then, inspired by article "Learning from Simulated and Unsupervised Images through Adversarial
Training", we want to add an attention to our model in p2p_channel.py and p2p_spatial.py.

![excitation](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/excitation.png)



And we have comparaison of PSNR and SSIM.

![psnr_compare](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/psnr_compare.png)![ssim_compare](/Users/lvchen/Downloads/3a/MAP583/GitHub/deraining/images/ssim_compare.png)

