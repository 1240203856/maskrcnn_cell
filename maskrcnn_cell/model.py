import torch
import torch.nn as nn
import torchvision
from models.detection.mask_rcnn import MaskRCNNPredictor,MaskRCNN
from models.detection.rpn import AnchorGenerator
import models


def get_instance_segmentation_model(bone='resnet50',attention=False):
    if bone=='mobilenet_v2':
        if attention==False:
            backbone = models.mobilenet_v2(pretrained=True,att=attention).features
        if attention==True:
            backbone = models.mobilenet_v2(pretrained=False,att=attention).features
        backbone.out_channels = 1280
    if bone=='googlenet':
        if attention==False:
            backbone = models.googlenet(pretrained=True)
        if attention==True:
            backbone = models.googlenet(pretrained=False)
        backbone.out_channels = 1024
    if bone=='densenet121':
        if attention==False:
            backbone = models.densenet121(pretrained=True,att=attention).features
        if attention==True:
            backbone = models.densenet121(pretrained=False,att=attention).features
        backbone.out_channels = 1024
    if bone=='resnet50':
        if attention==False:
            backbone = models.resnet50(pretrained=True,att=attention)
        if attention==True:
            backbone = models.resnet50(pretrained=False,att=attention)
        backbone.out_channels = 2048
    if bone=='shufflenet_v2_x1_0':
        if attention==False:
            backbone = models.shufflenet_v2_x1_0(pretrained=True)
        if attention==True:
            backbone = models.shufflenet_v2_x1_0(pretrained=False)
        backbone.out_channels = 1024
    if bone=='inception_v3':
        if attention==False:
            backbone = models.inception_v3()    #'InceptionOutputs' object has no attribute 'values'
        if attention==True:
            backbone = models.inception_v3()    #'InceptionOutputs' object has no attribute 'values'
        backbone.out_channels = 2048
    if bone=='squeezenet1_0':
        if attention==False:
            backbone = models.squeezenet1_0(pretrained=True).features
        if attention==True:
            backbone = models.squeezenet1_0(pretrained=False).features
        backbone.out_channels = 512

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)
    model = MaskRCNN(backbone,
                     num_classes=2,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler)
    return model