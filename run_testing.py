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

# #############
# Functions
# #############

def run_test(dataset,model_name,train_info,test_info,thres=0.001):
    '''
    run testing on a "dataset"

    :param dataset:
    :param model_name:
    :param train_info:
    :param test_info:
    :param thres: confidence for a bb
    :return:
    '''
    #pdb.set_trace()
    
    model_cfg_path = os.path.join(wd,'material','cfg','%s.cfg'%test_info)
    model_weights_path = os.path.join(wd,'material','yolo_models','%s'%train_info,'%s.weights'%model_name)
    meta_path = os.path.join(wd,'material','cfg','%s'%data_info)

    dataset_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '%s.txt'%dataset)
    
    # predict results
    predict_results_dir = os.path.join(wd,'results', 'predict', test_info)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    predict_results_dir = os.path.join(predict_results_dir, model_name)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    predict_results_dir = os.path.join(predict_results_dir,dataset)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    # --get testing data
    fp = open(dataset_file,'r')
    img_names = fp.readlines()
    fp.close()

    img_names_list = [x.strip() for x in img_names]

    img_paths = []
    img_labels_path = []
    for name in img_names_list:
        path = os.path.join(dataset_dir, 'JPEGImages','%s.jpg'%name)
        label_path = os.path.join(dataset_dir, 'dk_labels','%s.txt'%name)
        if not os.path.isfile(path):
            print ("file missing!")
            print (path)
            exit()
        if not os.path.isfile(label_path):
            print ("file missing!")
            print (label_path)
            exit()
        img_paths.append(path)
        img_labels_path.append(label_path)

    # --run testing
    #pdb.set_trace()

    # load yolo-model
    net = load_net(model_cfg_path, model_weights_path, 0)
    # load Experiment information
    meta = load_meta(meta_path)

    for idx,item in enumerate(img_names_list):

        print ('Processing '+item)

        result_txt_path = os.path.join(predict_results_dir,'%s.txt'%item)

        result_f = open(result_txt_path,'w')

        res = detect(net, meta, img_paths[idx],thresh=thres)

        im = cv2.imread(img_paths[idx])

        (im_h,im_w,im_c) = im.shape

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

            result_f.write('%d %f %f %f %f %f\n'%(cls,prob,x,y,w,h))

        result_f.close()

def run_test_on_testset(dataset,model_name,train_info,test_info,thres=0.001):
    '''
    run testing on a "dataset"
    :param dataset:
    :param model_name:
    :return:
    '''

    model_cfg_path = os.path.join(wd, 'material', 'cfg', '%s.cfg' % test_info)
    model_weights_path = os.path.join(wd, 'material', 'yolo_models', '%s' % train_info, '%s.weights' % model_name)
    meta_path = os.path.join(wd, 'material', 'cfg', '%s' % data_info)

    dataset_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '%s.txt'%dataset)

    # predict results
    predict_results_dir = os.path.join(wd,'results', 'predict', test_info)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    predict_results_dir = os.path.join(predict_results_dir, model_name)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    predict_results_dir = os.path.join(predict_results_dir,dataset)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    # --get testing data
    fp = open(dataset_file,'r')
    img_names = fp.readlines()
    fp.close()

    img_names_list = [x.strip() for x in img_names]

    img_paths = []
    img_labels_path = []
    for name in img_names_list:
        path = os.path.join(dataset_dir, 'JPEGImages','%s.jpg'%name)
        if not os.path.isfile(path):
            print ("file missing!")
            print (path)
            exit()

        img_paths.append(path)

    # --run testing
    #pdb.set_trace()
    net = load_net(model_cfg_path, model_weights_path, 0)
    meta = load_meta(meta_path)

    for idx,item in enumerate(img_names_list):

        print ('Processing '+item)

        result_txt_path = os.path.join(predict_results_dir,'%s.txt'%item)

        result_f = open(result_txt_path,'w')

        res = detect(net, meta, img_paths[idx],thresh=thres)

        im = cv2.imread(img_paths[idx])

        (im_h,im_w,im_c) = im.shape

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

            result_f.write('%d %f %f %f %f %f\n'%(cls,prob,x,y,w,h))

        result_f.close()

if __name__ == "__main__":
    # {(model_name,dataset_name),...,...}

    # prefix of yolo-model file (.weight)
    ds_prefix = 'yolo-voc-800_'
    # folder of yolo model files (default dir: ./yolo_models/)
    ds_train = 'missfresh-yolo-voc-800'
    # folder of predicting results (default dir: ./results/predict/)
    ds_test = 'missfresh-yolo-voc-800'


    # DataSets Type(train/val/test)
    #sets = ['val','train']
    sets = ['test']

    # checkpoints of models
    checkpoints = [15000]

    # Creating Dataset information
    DataSets = make_dataset(prefix=ds_prefix,train_info=ds_train,test_info=ds_test,sets=sets,iterations=checkpoints)

    #pdb.set_trace()

    # get predict results
    for ds in DataSets:
        model_name = ds[0]
        dataset_name = ds[1]
        train_info = ds[2]
        test_info = ds[3]

        #run_test_on_testset(dataset_name,model_name,train_info,test_info)

        if dataset_name=='test':
            #针对未标记的testSet
            sub_process = mp.Process(target=run_test_on_testset,args=(dataset_name,model_name,train_info,test_info))
        else:
            #针对已标记的trainSet/valSet
            sub_process = mp.Process(target=run_test, args=(dataset_name, model_name,train_info,test_info))
        sub_process.start()
        sub_process.join()
        sub_process.terminate()




