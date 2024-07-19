import os

import numpy as np
import torch

from classification import (Classification, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from utils.utils_metrics import CIFAR_evaluteTop1_5
import torchvision
import torchvision.transforms as transforms

#------------------------------------------------------#
#   CIFAR_path    CIFAR数据集的路径
#------------------------------------------------------#
CIFAR_path    = '/media/lht/LHT/code/datasets'
#------------------------------------------------------#
#   metrics_out_path        指标保存的文件夹
#------------------------------------------------------#
metrics_out_path        = "metrics_out"

#------------------------------------------------------#
#   CIFAR100 or CIFAR10
#------------------------------------------------------#
CIFAR = "CIFAR100"

class Eval_Classification(Classification):
    def detect_image(self, photo):        
        # #---------------------------------------------------------#
        # #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        # #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # #---------------------------------------------------------#
        # image       = cvtColor(image)
        # #---------------------------------------------------#
        # #   对图片进行不失真的resize
        # #---------------------------------------------------#
        # image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        # #---------------------------------------------------------#
        # #   归一化+添加上batch_size维度+转置
        # #---------------------------------------------------------#
        # image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            # photo   = torch.from_numpy(image).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
        return preds

if __name__ == "__main__":
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)
            
    classfication = Eval_Classification()
    
    #---------------------------------------------------------#
    #   读取CIFAR数据集
    #---------------------------------------------------------#
    if CIFAR == "CIFAR100":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR100(root=CIFAR_path, train=False, download=False, transform=transform)
        gen_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    elif CIFAR == "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root=CIFAR_path, train=False, download=False, transform=transform)
        gen_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    top1, top5, Recall, Precision = CIFAR_evaluteTop1_5(classfication, gen_val, metrics_out_path)
    print("top-1 accuracy = %.2f%%" % (top1*100))
    print("top-5 accuracy = %.2f%%" % (top5*100))
    print("mean Recall = %.2f%%" % (np.mean(Recall)*100))
    print("mean Precision = %.2f%%" % (np.mean(Precision)*100))
