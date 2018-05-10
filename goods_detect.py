#coding: utf-8

import os,sys
from darknet import *
#import cv2
import pdb
#from utils import *
import multiprocessing as mp

from conf import *
import time
import datetime


"""
API for MissFresh Project
Run object detection on images, and get results
"""

# client provided internal goods_id 
classes_id = ['image-p-qcshnt-268ml','image-p-xxwqaywsn-2h',
              'image-p-xxwxjnn-4h','image-p-kkkl-330ml',
              'image-p-hnwssgnyl-250ml*4','image-p-sdlwlcwt-4p',
              'image-p-mncnn250ml','image-p-wtnmc-250ml',
              'image-p-lfyswnc-280ml','image-p-hbahtkkwmyr-250ml',
              'gcht','ynhlg',
              'celxl-4','image-p-hyd-mnsnn-4h',
              'image-p-nfnfccz-300ml','image-p-yydhgc235ml',
              'image-p-mqtzyl-1h','image-p-nfsqcpyzlc-500ml',
              'image-p-blkqsyw-1b','image-p-szsj-350ml',
              'image-p-hbzllm-1h','image-p-tybhc-500ml',
              'image-p-nfdfsymlhc-500ml',
              'image-p-wqmrcptz-300ml',
              'image-p-Hbytfspg-5g',
              'image-big-p-scppg-2l'
             ]

class_id_blacklist = [10,11,12]

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


def load_image_from_url_no_cv2(img_url,flag=True):
    
    save_tmp_path = './tmp/img.jpg'
    abspath1 = os.path.abspath(save_tmp_path)
        # load from url
    urllib.urlretrieve(img_url,abspath1)
    
    if(flag):
        save_img_path = './images/'+time.strftime("%y_%m_%d", time.localtime())+'/'
        if not os.path.exists(save_img_path):
          os.mkdir(save_img_path)
        save_img =save_img_path+time.strftime("%H_%M_%S", time.localtime())+'_'+str(datetime.datetime.now().microsecond)+'.jpg'
        abspath2 = os.path.abspath(save_img)
        urllib.urlretrieve(img_url,abspath2)

    return abspath1

# #######################
# Goods Detect
# #######################
model_cfg_path = os.path.join(wd, 'material', 'cfg', 'missfresh-cls26-yolo-voc-800.cfg')
model_weights_path = os.path.join(wd, 'material', 'yolo_models', 'missfresh-mix-yolo-voc-800', 'yolo-voc-800_24000.weights')
meta_path = os.path.join(wd, 'material', 'cfg', '%s' % data_info)
# --init detector
net = load_net(model_cfg_path, model_weights_path, 0)
meta = load_meta(meta_path)

def goods_detect_urls(
        img_urls,
        yolo_cfg_path=model_cfg_path,
        yolo_weights_path=model_weights_path,
        good_info_path=meta_path,
        conf_thres=[0.7]
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

    for url in img_urls:

        det_result = []

        #_,im_path = load_image_from_url(url)
        im_path = load_image_from_url_no_cv2(url)

        im_path = os.path.abspath(im_path)

	      #im_path = './tmp/1.jpg'

        res = detect(net, meta, im_path, thresh=0.2)

        # parse result
        for line in res:
            cls_name = line[0]
            cls = classes.index(cls_name)
            prob = line[1]
            #bb = line[2]
            cls_id = classes_id[cls] 

            if cls not in class_id_blacklist:
                if(len(conf_thres)==1 and prob>=conf_thres[0]):
                    #print('%d %f'%(cls,prob))
                    det_result.append([cls_id,prob])
                elif(len(conf_thres)==len(classes) and prob>=conf_thres[cls]):
                    det_result.append([cls_id,prob])
                elif(prob<conf_thres[cls] or prob<conf_thres[0]):
                    continue
                else:
                    print 'error'

        goods_det_results_dict[url] = det_result
	
    #print goods_det_results_dict

    return goods_det_results_dict

def goods_detect_urls_yi_plus_local_test(
        img_urls,
        yolo_cfg_path=model_cfg_path,
        yolo_weights_path=model_weights_path,
        good_info_path=meta_path,
        conf_thres=[0.7]
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

    for url in img_urls:

        det_result = []

        if(url[0]!='.'):
          im_path = load_image_from_url_no_cv2(url,False)
        else:
          im_path=url

        im_path = os.path.abspath(im_path)

        res = detect(net, meta, im_path, thresh=0.2)
	
        cls_arr=[x+1 for x in range(26)]
        num_cls=[0 for x in range(26)]

        # parse result
        for line in res:
            cls_name = line[0]
            cls = classes.index(cls_name)
            prob = line[1]
	    print line[2]
            if cls not in class_id_blacklist:
                if(len(conf_thres)==1 and prob>=conf_thres[0]):
                    num_cls[cls]+=1
                    print cls_name
                    print prob
                elif(len(conf_thres)==len(classes) and prob>=conf_thres[cls]):
                    num_cls[cls]+=1
                    print cls_name
                    print prob
                elif(prob<conf_thres[cls] or prob<conf_thres[0]):
                    continue
                else:
                    print 'error'

                
        merg=[cls_arr,num_cls]
        merg=map(list,zip(*merg))
        print merg

if __name__ == "__main__":

    #urls = ['http://yijiaadplatform.dressplus.cn/AdPlatform/images/2018-04-12/ba575e54-3e22-11e8-a283-d0817abd9fdc.jpg','http://yijiaadplatform.dressplus.cn/AdPlatform/images/2018-04-12/bbc399cc-3e22-11e8-b2da-d0817abd9fdc.jpg','http://yijiaadplatform.dressplus.cn/AdPlatform/images/2018-04-12/c20a8b62-3e22-11e8-845b-d0817abd9fdc.jpg','http://yijiaadplatform.dressplus.cn/AdPlatform/images/2018-04-12/c2c53f52-3e22-11e8-9d13-d0817abd9fdc.jpg']
    #urls =['http://yijiaadplatform.dressplus.cn/AdPlatform/images/2018-04-12/ba575e54-3e22-11e8-a283-d0817abd9fdc.jpg']
    urls = ['./tmp/img1.jpg','./tmp/img2.jpg','./tmp/img3.jpg','./tmp/img4.jpg','./tmp/img5.jpg']
    each_thres=[.65,.65,.65,.65,.65,.65,.65,.65,.65,.65,.65,.65,.65,.65,.20,.65,.65,.65,.65,.50,.65,.65,.65,.65,.65,.65]
    goods_detect_urls_yi_plus_local_test(urls,conf_thres=each_thres)



