
import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import references.detection.transforms as T

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_dir=os.path.join(self.root,'Images')
        self.label_dir=os.path.join(self.root,'Labels')		
        self.image_list=[]
        for image in os.listdir(self.image_dir):
            self.image_list.append(image)
    def __len__(self):
       return len(self.image_list)
    def __getitem__(self, idx):
        image_name=os.path.join(self.image_dir,self.image_list[idx])
        label_name=os.path.join(self.label_dir,self.image_list[idx].split('.')[0]+'.npy')
        img = Image.open(image_name)
        mask = np.load(label_name)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])+1
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])+1 
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)			
        return img, target

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, : ,i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
#        boxes[i] = np.array([y1, x1, y2, x2])
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32, copy=False)

if __name__ =="__main__":
    dataset=PennFudanDataset(root='./data-ins/train/',transforms=T.Compose([T.ToTensor()]))

    loader = DataLoader(dataset,
                        batch_size=16,
                        shuffle=False,
                        num_workers=0)

    for i,image,label in enumerate(loader):
        print(image.shape)
