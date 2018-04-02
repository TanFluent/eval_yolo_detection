import os,sys
import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from conf import *

# #############
# Functions
# #############

colors = ['r','g','b','c']

def plot_bb(im,bb,cls,scores=[],thres=0.7,textSize=4,textThickness=8):
    h,w,c = im.shape

    for idx,box in enumerate(bb):
        if len(scores) < 1:
            pass
        else:
            if scores[idx]<thres:
                continue
            else:
                pass

        b_w = box[2]*w
        b_h = box[3]*h
        c_x = box[0]*w
        c_y = box[1]*h

        x1 = int(max([0, (c_x - 0.5 * b_w) ]))
        x2 = int(min([w, (c_x + 0.5 * b_w) ]))
        y1 = int(max([0, (c_y - 0.5 * b_h) ]))
        y2 = int(min([h, (c_y + 0.5 * b_h) ]))

        cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(im,'%s'%(classes[cls[idx]]),(x1,int(y1*1.1)),cv2.FONT_HERSHEY_SIMPLEX,textSize,(0,200,255),thickness=textThickness)

    return im,thres

def main_plot_bb(dataset,model_name,test_info):
    dataset_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '%s.txt' % dataset)
    f = open(dataset_file)
    img_names = f.readlines()
    f.close()

    img_names = [x.strip() for x in img_names]

    # --get predict results
    predict_results_first_lever_dir = os.path.join(wd,'results', 'predict', test_info)
    if not os.path.exists(predict_results_first_lever_dir):
        print("Folder not exist!")
        print predict_results_first_lever_dir
        exit()

    predict_results_second_lever_dir = os.path.join(predict_results_first_lever_dir, model_name)
    if not os.path.exists(predict_results_second_lever_dir):
        os.mkdir(predict_results_second_lever_dir)

    predict_results_dir = os.path.join(predict_results_second_lever_dir, dataset)
    if not os.path.exists(predict_results_dir):
        print("Folder not exist!")
        print predict_results_dir
        exit()

    # --get GT
    gt_dir = os.path.join(dataset_dir, 'dk_labels')

    # --get Images
    raw_img_dir = os.path.join(dataset_dir, 'JPEGImages')

    for name in img_names:
        print name
        #pdb.set_trace()
        #
        predict_file = open(os.path.join(predict_results_dir,'%s.txt'%name))
        pt_list = predict_file.readlines()
        predict_file.close()
        pt_list = [x.strip().split(' ') for x in pt_list]

        pred_boxes = [[float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in pt_list]
        pred_class_ids = [int(x[0]) for x in pt_list]
        pred_scores = [float(x[1]) for x in pt_list]
        #
        gt_file = open(os.path.join(gt_dir, '%s.txt' % name))
        gt_list = gt_file.readlines()
        gt_file.close()
        gt_list = [x.strip().split(' ') for x in gt_list]

        gt_boxes = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in gt_list]
        gt_class_ids = [int(x[0]) for x in gt_list]

        #
        raw_img_path = os.path.join(raw_img_dir,'%s.jpg'%name)
        im = cv2.imread(raw_img_path)

        #pdb.set_trace()
        im_gt = im.copy()
        plot_bb(im_gt,gt_boxes,gt_class_ids)
        im_gt = cv2.resize(im_gt,(im.shape[1]/2,im.shape[0]/2),interpolation=cv2.INTER_CUBIC)

        im_pred = im.copy()
        plot_bb(im_pred, pred_boxes, pred_class_ids, pred_scores,thres=0.2)
        im_pred = cv2.resize(im_pred, (im.shape[1] / 2, im.shape[0] / 2), interpolation=cv2.INTER_CUBIC)

        # --plot
        im_final = np.hstack((im_gt,im_pred))

        im_final = cv2.resize(im_final, (im_final.shape[1] / 2, im_final.shape[0] / 2), interpolation=cv2.INTER_CUBIC)

        #win_name = '%s-%s-%s' % (model_name, dataset_name, name)

        
        # --OPENCV plot
        win_name = '%s-%s' % (model_name, dataset)
        cv2.imshow(win_name,im_final)
        cv2.moveWindow(win_name,10,10)
        
        k = cv2.waitKey(0)
        if k==ord('q'):
            cv2.destroyAllWindows()
            break
        elif k==ord('c'):
            cv2.destroyWindow(win_name)


        # --matplotlib plot
        #plt.imshow(im_final)
        #plt.show()
        #pdb.set_trace()
        
        # --save image
        #save_dir = os.path.join(wd,'results', 'tmp')
        #save_path = os.path.join(save_dir,'%s.jpg'%name)
        #plt.savefig(save_path)
        #pdb.set_trace()

def main_plot_bb_on_testset(dataset,model_name,test_info):
    dataset_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '%s.txt' % dataset)
    f = open(dataset_file)
    img_names = f.readlines()
    f.close()

    img_names = [x.strip() for x in img_names]

    # --get predict results
    predict_results_first_lever_dir = os.path.join(wd, 'results', 'predict', test_info)
    if not os.path.exists(predict_results_first_lever_dir):
        print("Folder not exist!")
        print predict_results_first_lever_dir
        exit()

    predict_results_second_lever_dir = os.path.join(predict_results_first_lever_dir, model_name)
    if not os.path.exists(predict_results_second_lever_dir):
        os.mkdir(predict_results_second_lever_dir)

    predict_results_dir = os.path.join(predict_results_second_lever_dir, dataset)
    if not os.path.exists(predict_results_dir):
        print("Folder not exist!")
        print predict_results_dir
        exit()

    # --get Images
    raw_img_dir = os.path.join(dataset_dir, 'JPEGImages')

    for name in img_names:
        print name
        #pdb.set_trace()
        #
        predict_file = open(os.path.join(predict_results_dir,'%s.txt'%name))
        pt_list = predict_file.readlines()
        predict_file.close()
        pt_list = [x.strip().split(' ') for x in pt_list]

        pred_boxes = [[float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in pt_list]
        pred_class_ids = [int(x[0]) for x in pt_list]
        pred_scores = [float(x[1]) for x in pt_list]

        #
        raw_img_path = os.path.join(raw_img_dir,'%s.jpg'%name)
        im = cv2.imread(raw_img_path)

        #pdb.set_trace()

        im_pred = im.copy()
        plot_bb(im_pred, pred_boxes, pred_class_ids, pred_scores,thres=0.2,textSize=2,textThickness=4)
        im_pred = cv2.resize(im_pred, (im.shape[1] / 2, im.shape[0] / 2), interpolation=cv2.INTER_CUBIC)

        #m --plot
        im_final = im_pred

        win_name = '%s-%s' % (model_name, dataset_name)
        cv2.imshow(win_name,im_final)
        cv2.moveWindow(win_name,10,10)

        k = cv2.waitKey(0)
        if k==ord('q'):
            cv2.destroyAllWindows()
            break
        elif k==ord('c'):
            cv2.destroyWindow(win_name)

if __name__ == "__main__":

    # {(model_name,dataset_name),...,...}
    DataSets = [('yolo-voc-800_15000', 'test', 'missfresh-yolo-voc-800')]

    # get predict results
    for ds in DataSets:
        model_name = ds[0]
        dataset_name = ds[1]
        test_info = ds[2]

        if dataset_name=='test':
            main_plot_bb_on_testset(dataset_name, model_name,test_info)
        else:
            main_plot_bb(dataset_name, model_name,test_info)