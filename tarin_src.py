import argparse
from tqdm import tqdm
from utils import *
import network
from torchvision import transforms
from torch.utils.data import DataLoader
import itertools
import random
import matplotlib.pyplot as plt
import os, sys
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score,f1_score, roc_curve, auc
import sklearn.metrics as metrics


# Settings
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# Show images
def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow( (image * 255).astype(np.uint8))
        plt.axis('off')

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def lightings(args):
    ########## linghtings - training #########
    # model
    # netAN = network.AutoEncoder().cuda()
    # print(netAN)
    # trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16,max_epochs=args.max_epoch) #
    # tr = trainer.fit(netAN, iter_source, val_loader)
    # trainer.test(netAN,dataloaders=val_loader,ckpt_path='best')
    ########## linghtings - training #########
    return None


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    fig = plt.figure()
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    try:
        plt.colorbar()
    except:
        pass
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.set_size_inches(10, 10)
    plt.savefig('confuse.png')
    plt.show()

IOU = 0
def cnf(test_target_,test_prex_,x1):
    cnf_matrix = metrics.confusion_matrix(test_target_,test_prex_)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix, classes=["ng","good"])
    TP = metrics.confusion_matrix(test_target_,test_prex_, labels=[1, 0])[0, 0]
    FP = metrics.confusion_matrix(test_target_,test_prex_, labels=[1, 0])[1, 0]
    FN = metrics.confusion_matrix(test_target_,test_prex_, labels=[1, 0])[0, 1]
    TN = metrics.confusion_matrix(test_target_,test_prex_, labels=[1, 0])[1, 1]
    TP,FP,FN,TN = float(TP),float(FP),float(FN),float(TN)
    IOU =TP/(TP+FP+FN)
    mode = {0:"train",1:"test"}
    
    print("IOU : {:.4f}".format(IOU))
    ioux = open('iou_{}.txt'.format(mode[x1]), 'a')#os.path.join(args.log, 'log.txt')
    ioux.write("\n IOU:{:.4f}\n".format(IOU))
    ioux.close()
    return IOU

def cnf2(test_target_,test_prex_):
    cnf_matrix = metrics.confusion_matrix(test_target_,test_prex_)
    #print(cnf_matrix)
    #plot_confusion_matrix(cnf_matrix, classes=["good","ng"])
    TP = metrics.confusion_matrix(test_target_,test_prex_, labels=[1, 0])[0, 0]
    FP = metrics.confusion_matrix(test_target_,test_prex_, labels=[1, 0])[1, 0]
    FN = metrics.confusion_matrix(test_target_,test_prex_, labels=[1, 0])[0, 1]
    TN = metrics.confusion_matrix(test_target_,test_prex_, labels=[1, 0])[1, 1]
    TP,FP,FN,TN = float(TP),float(FP),float(FN),float(TN)
    IOU =TP/(TP+FP+FN)
    print("IOU : {:.4f}".format(IOU))
    return IOU

def plt_auc(y_test_label,predict_y):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')

    fpr, tpr, _ = roc_curve(y_test_label, predict_y)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = {}'.format(roc_auc))

    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Plot loss and accuracy
def plt_loss(epoch_train_recall,epoch_test_recal):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epoch_train_recall)
    ax1.plot(epoch_test_recal)
    ax1.set_title('model Recall')
    ax1.set_ylabel('Recall')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left') 
    #plt.show()
    # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss') 
    plt.savefig('Recall.png')
    plt.show()




# set save training log_list
reall_all=[]
train_pre =[]
train_tar = []
f1_list = []
iou_list = []
epoch_train_recall=[]
def train_source(args):
    # data loader
    dset_loaders = dataset_load(args)
    iter_source = dset_loaders["source_tr"]
    val_loader = dset_loaders['source_te']
    
    # model
    
    netF = network.ResNet_FE().cuda()
    netC = network.feat_classifier().cuda()
    optimizer = optim.SGD([{
        'params': netF.feature_layers.parameters(),
        'lr': args.lr
    }, {
        'params': netF.bottle.parameters(),
        'lr': args.lr 
    }, {
        'params': netF.bn.parameters(),
        'lr': args.lr 
    }, {
        'params': netC.parameters(),
        'lr': args.lr 
    }],#* 10
                          momentum=0.9,
                          weight_decay=0.0005,
                          nesterov=True)
    
    for epoch in range(args.max_epoch):
        netF.train()
        netC.train()
        
        for batch_idx, (input_source, label_source) in enumerate(iter_source):
            if (batch_idx) >=800:
                break
            if input_source.size(0) == 1:
                continue
            input_source, label_source = input_source.cuda(), label_source.cuda()

            output = netF(input_source)
            # print(netC)
            output = netC(output)
            # output =  nn.Softmax(dim=1)(output)
            loss = celoss(output, label_source)
            _, prex= torch.max(output.cpu(), 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # evl recall, f1-score, iou
            train_pre.extend(nn.Softmax(dim=1)(prex).cpu())                    # batch accuracy
            target_source2=label_source
            train_tar.extend(target_source2.cpu())          # batch label
            
            reall_x= recall_score(target_source2.cpu(),prex.cpu(),zero_division=1) # batch recall
            reall_all.append(reall_x)
            f1=f1_score(target_source2.cpu(),prex.cpu())    # batch f1-score
            f1_list.append(f1)
            iou_train2=cnf2(target_source2.cpu(),prex.cpu())# batch IOU
            iou_list.append(iou_train2)
            # save training log
            print("epoch: {:.4f}  ,loss: {:.4f}, f1-score: {:.4f}, iou: {:.4f}".format(epoch,loss,f1,iou_train2))
            log = open('./log/train_all.txt', 'a')
            log.write("res_recall:{:.4f} ,iou_train: {:.4f},f1-score: {:.4f}\n".format(reall_x,iou_train2,f1))
            log.close()
        
        # plot confuse matrix
        cnf(train_tar,train_pre,x1=0) 
        # plot ROC
        plt_auc(train_tar,train_pre)
        # calculate all-mean
        sumx=0
        for j in range(len(reall_all)):
            sumx+=reall_all[j]
        res_recall=sumx/len(reall_all)
        sumx=0
        for j in range(len(iou_list)):
            sumx+=(iou_list[j])
        iou_train=sumx/len(iou_list)
        sumx=0
        for j in range(len(f1_list)):
            sumx+=(f1_list[j])
        f1_train=sumx/len(f1_list)
        # print("total res_recall: ", res_recall)
        log = open('train.txt', 'a')
        log.write("res_recall:{:.4f} ,iou_train: {:.4f},f1-score: {:.4f}\n".format(res_recall,iou_train,f1_train))
        log.close()

        netF.eval()
        netC.eval()
        acc_s_tr, _ = cal_acc_(dset_loaders['source_te'], netF,netC)
        log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.4f}%'.format(
            args.dset, epoch + 1, args.max_epoch, acc_s_tr * 100)
        log_t = open('./log/train_acc.txt', 'a')
        log_t.write(log_str)
        log_t.close()
        epoch_train_recall.append(res_recall)
        # save model weight
        if acc_s_tr >= acc_init:
            acc_init = acc_s_tr
            best_netF = netF.state_dict()
            best_netC = netC.state_dict()
    torch.save(best_netF,  "./log/source_F.pt")
    torch.save(best_netC,  "./log/source_C.pt")


# set save testing log_list
reall_all2=[]
epoch_test_recal=[]
test_prex_ = []
test_target_ =[]
f1_test = []
iou_test = []

def test_target(args):
    dset_loaders = dataset_load(args)
    target_loader = dset_loaders['target']   # all target
    test_loader = dset_loaders['test']
    netF = network.ResNet_FE().cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()
    model_pathF = './log/source_F.pt'
    netF.load_state_dict(torch.load(model_pathF),False)
    model_pathC = './log/source_C.pt'
    netC.load_state_dict(torch.load(model_pathC),False)
    # netF=torch.nn.DataParallel(netF).to('cuda')  # multi-gpu
    # netC=torch.nn.DataParallel(netC).to('cuda')  # multi-gpu
    netF.eval()
    netC.eval()
    acc, Centropy = cal_acc_(dset_loaders['test'], netF, netC)
    log_str = 'Task: {}, TEST_Accuracy = {:.4f}%'.format(args.dset, acc * 100)
    log_t = open('./log/testall_acc.txt', 'a')
    log_t.write(log_str)
    log_t.close()

    # calculate  test data by batch
    iter_source2 = dset_loaders['test']
    for batch_idx, (inputs_source, labels_source) in enumerate(iter_source2):
        if inputs_source.size(0) == 1:
            continue
        inputs_source2, labels_source2 = inputs_source.cuda(), labels_source.cuda()
        output = netF(inputs_source2)
        output = netC(output)
        _, prex= torch.max(output.cpu(), 1)
        reall_xx=recall_score(prex.cpu(), labels_source2.cpu(),zero_division=1)
        reall_all2.append(reall_xx)
       #  print("test_recall:",reall_xx)
        test_prex_.extend(prex.cpu())
        test_target_.extend(labels_source2.cpu())
        f1=f1_score( labels_source2.cpu(),prex.cpu(),zero_division=1)
        f1_test.append(f1)
        iou_t=cnf2( labels_source2.cpu(),prex.cpu())
        iou_test.append(iou_t)
        print("batch_idx:{:.4f} , f1-score: {:.4f}, iou: {:.4f}".format(batch_idx,f1,iou_t))
        log = open('test_batchall.txt', 'a')
        log.write("\n res_recall: {:.4f},f1_test: {:.4f}, iou_test: {:.4f} \n".format(reall_xx,f1,iou_t))
        log.close()
    # calculate batch mean 
    sumx=0
    for j in range(len(reall_all2)):
        sumx+=reall_all2[j]
    test_recall=sumx/len(reall_all2)
    epoch_test_recal.append(test_recall)
    try:
        cnf(test_target_,test_prex_,x1=1)
    except:
        pass
    sumx=0
    for j in range(len(iou_test)):
        sumx+=(iou_test[j])
    iou_test2=sumx/len(iou_test)
    sumx=0
    for j in range(len(f1_test)):
        sumx+=(f1_test[j])
    f1_test2=sumx/len(f1_test)

    log = open('test_batch_mean.txt', 'a')
    log.write("\n res_recall: {:.2f},f1_test: {:.2f}, iou_test: {:.2f} \n".format(test_recall,f1_test2,iou_test2))
    log.close()









            
            



    
    

    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DataSet to Classification Task"
    )
    parser.add_argument('--s_set', type=str, default='source',help='source dataset for train') # dset
    parser.add_argument('--t_set', type=str, default='target',help='target dataset for train')  # dset
    parser.add_argument('--max_epoch',type=int,default=4000,help="number of epoch")
    parser.add_argument('--batch', type=int, default=10 )
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu_id',type=str, default='0')
    parser.add_argument('--seed', type=int, default=20223, help="random seed")
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--lr',
                        type=float,
                        default=0.000161,
                        help="learning rate")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    # train source
    train_source(args)
    # test traget
    test_target(args)
    try:
        plt_loss(epoch_train_recall,epoch_test_recal)
        IOU=cnf(test_target_,test_prex_,x1=1)
    except:
        pass