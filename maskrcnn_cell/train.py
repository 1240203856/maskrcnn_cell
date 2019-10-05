import torch
import torch.nn as nn
import torchvision
from models.detection.faster_rcnn import FastRCNNPredictor
from models.detection.mask_rcnn import MaskRCNNPredictor,MaskRCNN
import references.detection.utils as utils
from references.detection.engine import train_one_epoch, evaluate
from dataloader import PennFudanDataset
from PIL import Image
import os
import numpy as np
import matplotlib as mpl
import numpy
from models.detection.rpn import AnchorGenerator
import models
from model import get_instance_segmentation_model
from transform import get_transform

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# use the PennFudan dataset and defined transformations
dataset = PennFudanDataset(root='./patch_data/', transforms=get_transform(train=True))
dataset_test = PennFudanDataset(root='./patch_data_test/same_tissue/', transforms=get_transform(train=False))

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

# the dataset has two classes only - background and person
num_classes = 2

# get the model using the helper function
#bone: 'resnet50'/'mobilenet_v2'/'googlenet'/'densenet121'/'resnet50'/'shufflenet_v2_x1_0'/'inception_v3'/'squeezenet1_0'/
#attention：True/False
model = get_instance_segmentation_model(bone='resnet50',attention=False)
model.cuda()
model = nn.DataParallel(model)

# move model to the right device
#model.load_state_dict(torch.load('/disk2/yzy_cell/cell_model_densenet'))
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005,
                            momentum=0.9, weight_decay=0.0005)

# the learning rate scheduler decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=20,
                                               gamma=0.1)

# training
num_epochs = 200

def train():
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(),'/disk2/yzy_cell/cell_model_resnet50')

if __name__=='__main__':
    train()


