import numpy as np
import argparse
from PIL import Image
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])

def image_target(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])

def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        normalize
    ])

def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        print(len_)
        image = [(image_list[i].strip(),labels[i,:]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0],
            np.array([int(la) for la in val.split()[1:]]))
            for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1]))
            for val in image_list]
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs 
        self.transform = transform
        self.traget_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.traget_transform is not None:
            target = self.traget_transform(target)
        return img, target 
    
    def __len__(self):
        return len(self.imgs)

def dataset_load(args):
    # when we train the model, we use all training data set as 'source_tr'.
    # we can use limite training data to be training validation set as 'source_te'.
    # whine we test the model, we use all testing data set as 'target'.
    # we can use limite testing data to be quickly inferece, as like 'test'. 
    batch_size = args.batch
    train_listimage=[]
    test_listimage=[] 
    # tr_listimage=[]
    # tr_te_listimage=[]
    ss = args.s_set
    tt = args.t_set
    s_train = './data/ori_img/{}.txt'.format(ss)
    t_test = './data/ori_img/{}.txt'.format(tt)
    txt_s_train = open(s_train).readlines()
    txt_t_test = open(t_test).readlines()

    # set(limite) numbers of train dataset
    c = 0
    f_s_t = random.sample(txt_s_train,len(txt_s_train)) 
    for i in f_s_t: 
        if c >= 100:
            continue
        else:
            train_listimage.append(i)
            c+=1
    print("train_listimage",len(train_listimage))

    # set(limite) numbers of test dataset
    cccc=0
    f_t_t = random.sample(txt_t_test,len(txt_t_test))
    for i in f_t_t:
            if cccc >= 100:
                continue
            else:
                test_listimage.append(i)
                cccc+=1
    print("test_listimage",len(test_listimage))

    prep_dict = {}
    prep_dict['source'] = image_train()   # trainset
    prep_dict['target'] = image_target()  # testset
    prep_dict['val'] = image_test()       # validation

    train_source = ImageList(f_s_t, transform=prep_dict['source'])          # all train data
    test_source = ImageList(train_listimage, transform=prep_dict['source']) # limite train data
    train_target = ImageList(f_t_t, transform=prep_dict['target'])          # all test data
    test_target = ImageList(test_listimage, transform=prep_dict['target'])  # limite test data

    dset_loader = {}
    dset_loader['source_tr'] = DataLoader(train_source,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          drop_last=False,
                                          pin_memory=True) # last all train set
    dset_loader['source_te'] = DataLoader(test_source,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          drop_last=False,
                                          pin_memory=True) # last train val set
    dset_loader['target'] = DataLoader(train_target,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          drop_last=False,
                                          pin_memory=True) # last test set  
    dset_loader['test'] = DataLoader(test_target,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          drop_last=False,
                                          pin_memory=True) # last limite test set 
    return dset_loader      
                            




# tool 
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self,
                 num_classes,
                 epsilon=0.1,
                 use_gpu=True,
                 size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1,
            targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 -
                   self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (-targets * log_probs).mean(0).sum()
        else:
            loss = (-targets * log_probs).sum(1)
        return loss


def celoss(input_pre, target):
    output =  nn.Softmax(dim=1)(input_pre)
    ce = nn.CrossEntropyLoss()
    loss = ce(output,  target)
    return loss


def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def cal_acc_(loader, netF, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            output_f = netF.forward(inputs)  # a^t
            outputs=netC(output_f)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float( all_label.size()[0])
    #mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item() #self netropy
    mean_ent = celoss(predict, all_label)
    return accuracy, mean_ent   