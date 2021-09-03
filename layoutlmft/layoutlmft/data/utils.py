import torch

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeTransform, TransformList

def normalize_bboxes_docvqa(bboxes, width, height):
  norm_bboxes = []
  for box in bboxes:
    x1,y1,x2,y2,x3,y3,x4,y4 = box
    new_x1 = min([x1,x2,x3,x4])
    new_x2 = max([x1,x2,x3,x4])
    new_y1 = min([y1,y2,y3,y4])
    new_y2 = max([y1,y2,y3,y4])
    assert new_x2 >= new_x1
    assert new_y2 >= new_y1
    new_x1 = int(1000 * (new_x1 / width))
    new_x2 = int(1000 * (new_x2 / width))
    new_y1 = int(1000 * (new_y1 / height))
    new_y2 = int(1000 * (new_y2 / height))
    norm_bboxes.append([new_x1,new_y1,new_x2,new_y2])
  return norm_bboxes[0]

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]


def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


def load_image(image_path):
    image = read_image(image_path, format="BGR")
    h = image.shape[0]
    w = image.shape[1]
    img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224)])
    image = torch.tensor(img_trans.apply_image(image).copy()).permute(2, 0, 1)  # copy to make it writeable
    return image, (w, h)
