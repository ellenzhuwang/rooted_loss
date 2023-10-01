# -*- coding: utf-8 -*-
'''
Training various of neural networks with Rooted Logistic Objectives
Written by Ellen Wang 

'''


from torchvision import datasets, transforms, models
import torch
import os


#loading for food-101 dataset

food_data_dir = 'food-101'
def foodDataset(data_dir=food_data_dir,batch_size = bs,Transformation=False,resize=224,smallset=False):

	train_transforms = transforms.Compose([transforms.RandomRotation(30),
		                               transforms.Resize((resize,resize)),
		                               transforms.RandomHorizontalFlip(),
		                               transforms.ToTensor(),
		                               transforms.Normalize([0.485, 0.456, 0.406],
		                                                    [0.229, 0.224, 0.225])])


	test_transforms = transforms.Compose([transforms.Resize((resize,resize)),
		                              #transforms.CenterCrop(resize),
		                              transforms.ToTensor(),
		                              transforms.Normalize([0.485, 0.456, 0.406],
		                                                   [0.229, 0.224, 0.225])])
	if not Transformation:
		Train_set=datasets.Food101(data_dir,split="train",transform=transforms.ToTensor(),download=False)
		Test_set=datasets.Food101(data_dir,split="test",transform=transforms.ToTensor(),download=False)
	else:
		Train_set=datasets.Food101(data_dir,split="train",transform=train_transforms,download=True)
		Test_set=datasets.Food101(data_dir,split="test",transform=test_transforms,download=True)
        
	if not smallset:  
		#Trainloader = torch.utils.data.DataLoader(Train_set, batch_size=batch_size, shuffle=True)
		#Testloader=torch.utils.data.DataLoader(Test_set, batch_size=batch_size,shuffle=False)
		return Train_set,Test_set
	else:
		Test_set, _ = torch.utils.data.random_split(Test_set, [5250, 20000])
		#Testloader=torch.utils.data.DataLoader(Test_set, batch_size=batch_size,shuffle=False)
		return Test_set


# Prepare dataset
id_dict = {}
tiny_data_dir = 'tiny-imagenet-200'
for i, line in enumerate(open('tiny-imagenet-200/wnids.txt', 'r')):
  id_dict[line.replace('\n', '')] = i

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.id_dict[img_path.split('/')[7]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label






