import torch
import torch.nn as nn
import torchvision
from models.detection.faster_rcnn import FastRCNNPredictor
from models.detection.mask_rcnn import MaskRCNNPredictor
import references.detection.utils as utils
from references.detection.engine import train_one_epoch, evaluate
from dataloader import PennFudanDataset
from PIL import Image
import os
import numpy as np
import matplotlib as mpl
import numpy
from color import pre_to_img
from transform import get_transform
from model import get_instance_segmentation_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# use the PennFudan dataset and defined transformations
dataset_test = PennFudanDataset(root='./patch_data_test/same_tissue/', transforms=get_transform(train=False))

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

# put the model in evaluation mode

def get_counts(sequence):
    counts={}
    for i in range(sequence.shape[0]):
        for x in sequence[i]:
            if x in counts:
                counts[x] += 1
            else: 
                counts[x] = 1
    return counts

# the dataset has two classes only - background and person
num_classes = 2

# get the model using the helper function
#bone: 'resnet50'/'mobilenet_v2'/'googlenet'/'densenet121'/'resnet50'/'shufflenet_v2_x1_0'/'inception_v3'/'squeezenet1_0'/
model = get_instance_segmentation_model(bone='densenet121',attention=False)
model.cuda()
model = nn.DataParallel(model)

# move model to the right device
model.to(device)

def test(img,m):
    with torch.no_grad():
        prediction = model([img.to(device)])

    Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).save('./eval/image/'+str(m)+'.png')

    mask=prediction[0]['masks'].mul(255).byte().cpu().numpy()
    mask=mask.squeeze()
    mask=mask/255
    mask=mask>0.7
    mask=mask*1

#    print(get_counts(mask[1]))

    score=prediction[0]['scores'].mul(100).byte().cpu().numpy()
    score=score.squeeze()
    score=score/100
#    print('score:',score)
    num_score=score.shape[0]

    mask0=np.zeros((mask.shape[1],mask.shape[2]))
    masks=[]
    masks.append(mask0)
    for i in range(num_score):
        if score[i]>0.7:
            mask[i]=mask[i]*(i+1)
            masks.append(mask[i])

    masks=np.array(masks)
#    print('masks.shape',masks.shape)
    masks=np.argmax(masks,0)

    masks=np.uint8(masks)

    Image.fromarray(masks).save('./eval/label/'+str(m)+'.png')

    img=Image.open('./eval/label/'+str(m)+'.png')
    image=np.array(img)
    image=pre_to_img(image)
    image.save('./eval/label-color/'+str(m)+'.png')
	
if __name__=='__main__':
    model.load_state_dict(torch.load('/disk2/yzy_cell/cell_model_densenet'))
    model.eval()
    # pick one image from the test set
    for i in range(len(dataset_test)):
        print(i)
        img, _ = dataset_test[i]
        test(img,i)