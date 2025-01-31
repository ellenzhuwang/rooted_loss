# Optimizing Neural Network Training and Quantization with Rooted Logistic Objectives (AISTATS 2025) :rocket:

First-order methods are widely employed for training neural networks that are used in practical applications. For classification of input features, Cross-Entropy based loss functions are often preferred since they are differentiable everywhere. Recent optimization results show that the convergence properties of first-order methods such as gradient descent are intricately tied to the separability of datasets and the induced loss landscape. We introduce Rooted Logistic Objectives (RLO) to improve practical convergence behavior with benefits for downstream tasks. We show that our proposed loss satisfies strict convexity properties and has better condition number properties that will benefit practical implementations. To evaluate our proposed RLO, we compare its performance on various classification benchmarks. Our results illustrate that training procedure converges faster with RLO in many cases. Furthermore, on two downstream tasks viz., post-training quantization and finetuning on quantized space, we show that it is possible to ensure lower performance degradation while using reduced precision for sequence prediction tasks in large language models over state of the art methods.

## 📜 Citation
If you find this project useful, please give us a star and cite:

```
@inproceedings{wang2025optimizing,
  title={Optimizing Neural Network Training and Quantization with Rooted Logistic Objectives},
  author={Wang, Zhu and Veluswami, Praveen Raj and Mishra, Harsh and Ravi, Sathya N},
  booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
  year={2025},
  url={https://openreview.net/forum?id=g5ml9INmja}
}
```
```
@article{wang2023accelerated,
  title={Accelerated Neural Network Training with Rooted Logistic Objectives},
  author={Wang, Zhu and Veluswami, Praveen Raj and Mishra, Harsh and Ravi, Sathya N},
  journal={arXiv preprint arXiv:2310.03890},
  year={2023}
}
```
## 🧠 Deep neural networks with RLO

### Datasets support:
- [x] CIFAR-10
- [x] CIFAR-100
- [x] Tiny-ImageNet
- [x] Food-101
- [x] ImageNet1k

More coming...
### Vision Models support:
- [x] VGG
- [x] ResNet (18, 34, 50, 101)
- [x] ViT (small, base, large)
- [x] vit_timm for finetuning
- [x] CaiT
- [x] Swin (base)

### LLM Models quantization support:
- [x] OPT
- [x] Llama2
- [x] Llama3
      
More coming...
### Loss function support:
- [x] cross entropy
- [x] focal
- [x] **RLO** :heart_eyes:

## How to use RLO in your NN training/quantization: 
Default settings: dataset: cifar10, net: ViT, loss: root, epochs:200, k:3, m:3
```
python train.py
```
Other settings example:
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


