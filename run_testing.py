import os,sys
from darknet import *
import cv2
import pdb
from utils import *
import multiprocessing as mp

# working dir
wd = '/home/tfl/workspace/project/YI/goodsid/'
# dataSets dir
dataset_dir = '/home/tfl/workspace/dataSet/GoodsID'
# classes
classes = ['beer','beverage','instantnoodle','redwine','snack','springwater','yogurt']

# #############
# Functions
# #############

def run_test(dataset,model_name,train_info,test_info,thres=0.001):
    '''
    run testing on a "dataset"
    :param dataset:
    :param model_name:
    :return:
    '''

    model_cfg_path = os.path.join(wd,'cfg','%s.cfg'%test_info)
    model_weights_path = os.path.join(wd,'%s'%train_info,'%s.weights'%model_name)
    meta_path = os.path.join(wd,'cfg','goodid.data')

    dataset_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '%s.txt'%dataset)

    # predict results
    predict_results_dir = os.path.join(dataset_dir, 'predict', test_info)
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

def run_test_on_mix(dataset,model_name,train_info,test_info,thres=0.001):
    '''
    run testing on a "dataset"
    :param dataset:
    :param model_name:
    :return:
    '''

    model_cfg_path = os.path.join(wd,'cfg','%s.cfg'%test_info)
    model_weights_path = os.path.join(wd,train_info,'%s.weights'%model_name)
    meta_path = os.path.join(wd,'cfg','goodid.data')

    dataset_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '%s.txt'%dataset)

    # predict results
    predict_results_dir = os.path.join(dataset_dir, 'predict', test_info)
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
        path = os.path.join(dataset_dir, 'images-raw','mix','%s.jpg'%name)
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
    #DataSets = [('yolo-voc_40000', 'val'),('yolo-voc_40000', 'train')]
    ds_prefix = 'yolo-voc-800-multiscale_'
    ds_train = 'nl_models_4w_lr_0.001_800x800_multiscale'
    ds_test = 'nl-yolo-voc-800-multiscale'

    #DataSets = make_dataset(prefix=ds_prefix, train_info=ds_train, test_info=ds_test)
    DataSets = make_dataset(prefix=ds_prefix,train_info=ds_train,test_info=ds_test,sets=['val','train'],iterations=[36000])
    #DataSets = [('yolo_40000','test',ds_train,ds_test)]

    # get predict results
    for ds in DataSets:
        model_name = ds[0]
        dataset_name = ds[1]
        train_info = ds[2]
        test_info = ds[3]

        if dataset_name=='test':
            sub_process = mp.Process(target=run_test_on_mix,args=(dataset_name,model_name,train_info,test_info))
        else:
            sub_process = mp.Process(target=run_test, args=(dataset_name, model_name,train_info,test_info))
        sub_process.start()
        sub_process.join()
        sub_process.terminate()




