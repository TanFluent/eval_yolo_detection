#coding: utf-8

import os,sys
from darknet import *
import cv2
import pdb
from utils import *
import multiprocessing as mp

from conf import *


"""
Run object detection on images, and get results
"""

# #######################
# Util
# #######################
import urllib
import numpy as np

def load_image_from_url(img_url):
    savepath = './tmp/img.jpg'

    # load from url
    data = urllib.urlopen(img_url).read()

    # cv2 format transfer
    img = np.asarray(bytearray(data),dtype="uint8")
    img = cv2.imdecode(img,cv2.IMREAD_COLOR)

    # save image to ./tmp
    abspath = os.path.abspath(savepath)
    if os.path.exists(abspath):
        os.system('rm %s'%abspath)

    cv2.imwrite(abspath,img)

    return img,abspath

# #######################
# Goods Detect
# #######################
model_cfg_path = os.path.join(wd, 'material', 'cfg', 'missfresh-yolo-voc-800.cfg')
model_weights_path = os.path.join(wd, 'material', 'yolo_models', 'missfresh-mix-yolo-voc-800', 'yolo-voc-800_2000.weights')
meta_path = os.path.join(wd, 'material', 'cfg', '%s' % data_info)

def goods_detect_urls(
        img_urls,
        yolo_cfg_path=model_cfg_path,
        yolo_weights_path=model_weights_path,
        good_info_path=meta_path,
        conf_thres=0.2
):
    """
    Performing goods detection on online images.

    :param img_urls:
    :param yolo_cfg_path:
    :param yolo_weights_path:
    :param good_info_path:
    :param conf_thres:
    :return:
    """
    goods_det_results_dict = {}

    # --init detector
    #pdb.set_trace()
    net = load_net(yolo_cfg_path, yolo_weights_path, 0)
    meta = load_meta(good_info_path)

    #res = detect(net, meta, img_paths[idx], thresh=thres)

    for url in img_urls:

        det_result = []

        _,im_path = load_image_from_url(url)
        im_path = os.path.abspath(im_path)
        res = detect(net, meta, im_path, thresh=conf_thres)

        # parse result
        for line in res:
            cls_name = line[0]
            cls = classes.index(cls_name)
            prob = line[1]
            bb = line[2]
            # convert bb

            x = bb[0] / im_w
            y = bb[1] / im_h
            w = bb[2] / im_w
            h = bb[3] / im_h

            det_result.append([cls,prob,x,y,w,h])

        goods_det_results_dict[url] = det_result

    #print goods_det_results_dict

    return goods_det_results_dict

if __name__ == "__main__":
    urls = ['http://mall8.qiyipic.com/mall/20170605/fc/2e/mall_5934fa87ad8c1223bb3bfc2e_1x1.jpg']

    goods_detect_urls(urls)


