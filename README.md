# Optimizing Neural Network Training and Quantization with Rooted Logistic Objectives :rocket:

Many neural networks deployed in real-world scenarios are trained using cross-entropy based loss functions. From the optimization perspective, it is known that the behavior of first-order methods such as gradient descent crucially depends on the separability of datasets. We focus on the landscape design of the logistic function and derive a novel sequence of {\em strictly} convex functions that are at least as strict as logistic loss. The minimizers of these functions coincide with those of the minimum norm solution wherever possible. The strict convexity of the derived function can be extended to finetune state-of-the-art models and applications. In empirical experimental analysis, we apply our proposed rooted logistic objective to multiple deep models on various of classification benchmarks. Our results illustrate that training with rooted loss function converges faster and gains performance improvements. Furthermore, we explore the robustness of this rooted loss function to provide enhanced quantization operation for sequence prediction tasks in large language models. By integrating a better-conditioned loss landscape, we facilitate post-training quantization and finetuning with quantizer with our proposed loss, which ensures minimal performance degradation with reduced precision. Additional applications of our novel rooted loss function include generative modeling based downstream applications, such as finetuning the StyleGAN model with the rooted loss.

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
## LLMs Quantization with RLO:

Finetune OPT with RLO:
Default settings: dataest: wikitext2, model: facebook/opt_125m. epochs: 3, k:5, m:5
```
pyhon ft_opt.py
```
Quantization:
```
CUDA_VISIBLE_DEVICES=0 python opt.py model_name wikitext2 --wbits 2 --quant ldlqRG --incoh_processing --save save_path
```

## GAN with RLO:
We use the official PyTorch implementation of the StyleGAN2-ADA from https://github.com/NVlabs/stylegan2-ada-pytorch/ to demonstrate the results of using the rooted loss replacing the original cross-entropy loss.
Clone the official StyleGAN2-ADA code using the below command.
```
https://github.com/NVlabs/stylegan2-ada-pytorch.git
```
Steps to implement our experiments.
1. Please follow the instructions in their documentation to prepare the dataset. Store 'FFHQ' and 'Stanford Dogs' images in their respective folders in './train_dataset'
2. Make appropriate changes to the file 'loss.py' as given in our repository.
3. Change the values of the variables 'kparam' and 'ls' as per requirement. Default settings: kparam=2 ; ls='rlo'
4. Refer to the file 'commands.txt' to find example commands for respective tasks/experiments.


