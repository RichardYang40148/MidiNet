This repository is still under preperation, we plan to release a trained model and a refined version of MidiNet(v1) around August. However, if anyone want to have a first look for the implementation, you can find the source code in this directory.

This repository contains the source code of [MdidNet : A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation](https://arxiv.org/abs/1703.10847)

<img src="network_structure.png" height="350">

## Notes

This is a slightly modified version of the model that we presented in the above paper, you can find notations in the code if the parameters differ from the paper.

These scripts are refer to [A tensorflow implementation of "Deep Convolutional Generative Adversarial Networks](https://github.com/carpedm20/DCGAN-tensorflow)

Thanks to Taehoon Kim / @carpedm20 for releasing such a decent DCGAN implementaion

## Instructions

The repository contains one trained model, which is  trained under only 50496 midi bars(augmented from 4208 bars), so the generator might sounds not so "creative".

It's quite fun to use Tencorboard to check out the model's training process: 
```
tensorboard --logdir=log/
```
You can check out the loss in the training, and the embedding visulizations of real and fake datas.
<img src="embedding.png" height="350">

To train by your own dataset:
```
1. change line 134-136 to your data path
2. run main.py --is_train True
```
## Requirements
[Tensorflow 0.11.0](https://github.com/tensorflow/tensorflow/tree/r0.11)

[python-midi](https://github.com/vishnubob/python-midi)
