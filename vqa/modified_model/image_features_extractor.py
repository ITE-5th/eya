import os

os.environ['GLOG_minloglevel'] = '3'
import sys
import time

from file_path_manager import FilePathManager

sys.path.insert(0, FilePathManager.resolve('vqa/modified_model/caffe/python/'))
sys.path.insert(0, FilePathManager.resolve('vqa/modified_model/lib/'))
sys.path.insert(0, FilePathManager.resolve('vqa/modified_model/tools/'))
import caffe
import cv2
import numpy as np
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test import im_detect, _get_blobs

MIN_BOXES = 36
MAX_BOXES = 36
data_path = FilePathManager.resolve('vqa/modified_model/data/genome/1600-400-20')
classes = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

attributes = ['__no_attribute__']
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())
caffe.set_device(0)
use_gpu = True
if use_gpu:
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()
cfg_from_file(FilePathManager.resolve('vqa/modified_model/experiments/cfgs/faster_rcnn_end2end_resnet.yml'))
weights = FilePathManager.resolve(
    'vqa/modified_model/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel')
prototxt = FilePathManager.resolve(
    'vqa/modified_model/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt')
net = caffe.Net(prototxt, caffe.TEST, weights=weights)


class ImageFeaturesExtractor:

    @staticmethod
    def extract_from_image(image, conf_thresh=0.2):
        caffe.set_mode_cpu()
        im = image
        image_h, image_w, _ = im.shape
        scores, boxes, rois = im_detect(net, im)
        blobs, im_scales = _get_blobs(im, None)
        cls_boxes = rois[:, 1:5] / im_scales[0]
        cls_prob = net.blobs['cls_prob'].data
        pool5 = net.blobs['pool5_flat'].data
        max_conf = np.zeros((rois.shape[0]))
        for cls_ind in range(1, cls_prob.shape[1]):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(nms(dets, cfg.TEST.NMS))
            max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
        keep_boxes = np.where(max_conf >= conf_thresh)[0]
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
        boxes, features = cls_boxes[keep_boxes], pool5[keep_boxes]
        boxes = boxes.reshape(36, -1)
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        scaled_width = box_width / image_w
        scaled_height = box_height / image_h
        scaled_x = boxes[:, 0] / image_w
        scaled_y = boxes[:, 1] / image_h
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = np.concatenate(
            (scaled_x,
             scaled_y,
             scaled_x + scaled_width,
             scaled_y + scaled_height,
             scaled_width,
             scaled_height),
            axis=1)
        return spatial_features, features

    @staticmethod
    def extract_from_path(image_path, conf_thresh=0.2):
        return ImageFeaturesExtractor.extract_from_image(cv2.imread(image_path), conf_thresh)


if __name__ == '__main__':
    result = ImageFeaturesExtractor.extract_from_path(FilePathManager.resolve("face/test_images/zaher.jpg"))
    time.sleep(10)
