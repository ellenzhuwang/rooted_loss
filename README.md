# Accelerated Neural Network Training with Rooted Logistic Objectives :rocket:
Many neural networks deployed in the real world scenarios are trained using cross entropy based loss functions. From the optimization perspective, it is known that the behavior of first order methods such as gradient descent crucially depend on the separability of datasets. In fact, even in the most simplest case of binary classification, the rate of convergence depends on two factors: 1. condition number of data matrix, and 2. separability of the dataset. With no further pre-processing techniques such as over-parametrization, data augmentation etc., separability is an intrinsic quantity of the data distribution under consideration. We focus on the landscape design of the logistic function and derive a novel sequence of strictly convex functions that are at least as strict as logistic loss. The minimizers of these functions coincide with those of the minimum norm solution wherever possible. The strict convexity of the derived function can be extended to finetune state-of-the-art models and applications. In empirical experimental analysis, we apply our proposed rooted logistic objective to multiple deep models, e.g., FCNs and transformers, on various of classification benchmarks. Our results illustrate that training with rooted loss function is **converged faster** and **gains performance improvements**. Furthermore, we illustrate applications of our novel rooted loss function in generative modeling based downstream applications, such as finetuning StyleGAN model with the rooted loss. 
## Regression with RLO

## Deep neural networks with RLO

### Datasets support:
- [x] CIFAR-10
- [x] CIFAR-100
- [x] Tiny-ImageNet
- [x] Food-101

More coming...
### Models support:
- [x] VGG
- [x] ResNet (18, 34, 50, 101)
- [x] ViT (small, base, large)
- [x] vit_timm for finetuning
- [x] CaiT
- [x] Swin (base)

More coming...
### Loss function support:
- [x] cross entropy
- [x] focal
- [x] **root** :heart_eyes:

### Usage examples: 
Default settings: dataset: cifar10, net: ViT, loss: root, epochs:200, k:3, m:3
```
python train.py
```
```
python train.py --dataset cifar100 --net Swin --k 8 --m 10
```
 
## GAN with RLO
