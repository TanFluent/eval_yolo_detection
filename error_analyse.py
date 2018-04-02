#coding: utf-8

import os,sys
import cv2
import numpy as np
from utils import *
import evaluation as my_eval

import pdb

from conf import *

# #############
# Functions
# #############

colors = ['r','g','b','c']

def get_eval_results(gt_boxes, gt_class_ids,
               pred_boxes, pred_class_ids, pred_scores,
               iou_threshold=0.5):
    '''

    :param gt_boxes:
    :param gt_class_ids:
    :param pred_boxes:
    :param pred_class_ids:
    :param pred_scores:
    :param iou_threshold:
    :return:
    '''

    ## --Step 1:Get sorted(by predicting confidence) ious between GT and Predicting results
    gt_boxes = my_eval.trim_zeros(gt_boxes)
    pred_boxes = my_eval.trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    indices = np.argsort(pred_scores)[::-1]  # top2bottom
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = my_eval.compute_overlaps(pred_boxes, gt_boxes)

    ## --Step 2: get error prediction
    fp_pred_mismatch_cls_index = []
    fp_pred_mismatch_bb_index = []
    fp_pred_mismatch_bb_redun_index = []
    fn_gt_index = []

    # Loop through ground truth boxes and find matching predictions
    match_count = 0
    pred_match = np.zeros([pred_boxes.shape[0]])
    gt_match = np.zeros([gt_boxes.shape[0]])  # tags:if a gt is matched,set its tag to 1;
    # Loop prediction
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        sorted_ixs = np.argsort(overlaps[i])[::-1]  # top2bottom
        # Loop GT
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] == 1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                break

        # get False Positive
        if pred_match[i] == 0:
            for j in sorted_ixs:
                iou = overlaps[i, j]
                #print ('iou:%f,i:%f'%(iou,i))
                if iou>iou_threshold:
                    if pred_class_ids[i] == gt_class_ids[j]:
                        fp_pred_mismatch_bb_redun_index.append(indices[i])
                    else:
                        fp_pred_mismatch_cls_index.append(indices[i])

            if indices[i] not in fp_pred_mismatch_cls_index and indices[i] not in fp_pred_mismatch_bb_redun_index:
                fp_pred_mismatch_bb_index.append(indices[i])
    #pdb.set_trace()
    #fp_pred_mismatch_bb_index = list(set(fp_pred_mismatch_bb_index))

    # get False Negtive
    fn_gt_index = np.where(gt_match==0)[0]

    ## --Step 3:Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match).astype(np.float32) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

    return precisions,recalls,fn_gt_index,fp_pred_mismatch_cls_index,fp_pred_mismatch_bb_index,fp_pred_mismatch_bb_redun_index

def main_error_analyse(dataset,model_name,test_info,conf_thres=0.2,should_show=False):
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

    avg_precise = 0.0
    avg_recall = 0.0

    tot_fn = 0
    tot_fp_cls = 0
    tot_fp_bb = 0
    tot_fp_redun = 0

    tot_pt_bb_num = 0
    tot_gt_bb_num = 0

    for name in img_names:
        #pdb.set_trace()
        # --get prediction
        predict_file = open(os.path.join(predict_results_dir,'%s.txt'%name))
        pt_list = predict_file.readlines()
        predict_file.close()
        pt_list = [x.strip().split(' ') for x in pt_list]

        pred_boxes = [[float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in pt_list]
        pred_class_ids = [int(x[0]) for x in pt_list]
        pred_scores = [float(x[1]) for x in pt_list]
        # --get ground truth
        gt_file = open(os.path.join(gt_dir, '%s.txt' % name))
        gt_list = gt_file.readlines()
        gt_file.close()
        gt_list = [x.strip().split(' ') for x in gt_list]

        gt_boxes = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in gt_list]
        gt_class_ids = [int(x[0]) for x in gt_list]

        # --get overlaps
        gt_boxes = np.array(gt_boxes)
        gt_class_ids = np.array(gt_class_ids)
        pred_boxes = np.array(pred_boxes)
        pred_class_ids = np.array(pred_class_ids)
        pred_scores = np.array(pred_scores)

        # convert bb format to (x1,y1,x2,y2)
        gt_boxes_convert = convert_bb_format(gt_boxes)
        pred_boxes_convert = convert_bb_format(pred_boxes)

        # only preserve large confidence scores BB
        pred_scores_idx = np.where(pred_scores > conf_thres)
        pred_scores = pred_scores[pred_scores_idx]
        pred_class_ids = pred_class_ids[pred_scores_idx]
        pred_boxes_convert = pred_boxes_convert[pred_scores_idx]
        pred_boxes = pred_boxes[pred_scores_idx]

        _,_,fn_gt_idx,fp_mismatch_cls,fp_mismatch_bb,fp_redun_bb = get_eval_results(
            gt_boxes_convert,
                gt_class_ids,
                pred_boxes_convert,
                pred_class_ids,
                pred_scores
        )

        precise = 1-(len(fp_mismatch_cls)+len(fp_mismatch_bb)+len(fp_redun_bb))*1.0/len(pred_boxes)*1.0
        recall = 1-len(fn_gt_idx)*1.0/len(gt_boxes)*1.0

        tot_fn = tot_fn + len(fn_gt_idx)
        tot_fp_bb = tot_fp_bb + len(fp_mismatch_bb)
        tot_fp_cls = tot_fp_cls + len(fp_mismatch_cls)
        tot_fp_redun = tot_fp_redun + len(fp_redun_bb)

        tot_pt_bb_num = tot_pt_bb_num + len(pred_boxes)
        tot_gt_bb_num = tot_gt_bb_num + len(gt_boxes)

        print("#%s -precise:%f  -recall:%f" % (name,precise, recall))
        print("%s -gt:%d  -pred:%d" % (name, len(gt_boxes), len(pred_boxes)))
        print("%s -fn:%d  -fp:%d" % (name, len(fn_gt_idx), (len(fp_mismatch_cls)+len(fp_mismatch_bb)+len(fp_redun_bb))))

        avg_precise = avg_precise + precise
        avg_recall = avg_recall + recall

        #pdb.set_trace()

        if should_show:
            # --get source image
            raw_img_path = os.path.join(raw_img_dir,'%s.jpg'%name)
            im = cv2.imread(raw_img_path)
            # plot gt
            im = plot_bb_on_img(im, gt_boxes, (255, 255, 255), info='',textThickness=1)
            # plot fn
            im = plot_bb_on_img(im,gt_boxes[fn_gt_idx],(255,255,255),info='fn',textThickness=8,bbThickness=8)
            # plot fp_cls
            im = plot_bb_on_img(im, pred_boxes[fp_mismatch_cls], (255, 0, 0), info='fp_cls',textThickness=4,bbThickness=8)
            # plot fp_bb
            im = plot_bb_on_img(im, pred_boxes[fp_mismatch_bb], (0, 255, 0), info='fp_bb', textThickness=4,bbThickness=8)
            # plot fp_redun
            im = plot_bb_on_img(im, pred_boxes[fp_redun_bb], (0, 0, 255), info='fp_redun', textThickness=4,bbThickness=8)

            cv2.putText(im,'-precise:%f -recall:%f'%(precise,recall),(10,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), thickness=4)

            im_resize = im.copy()
            im_resize = cv2.resize(im_resize, (int(im.shape[1] / 4), int(im.shape[0] / 4)), interpolation=cv2.INTER_CUBIC)

            win_name = '%s-%s' % (model_name, dataset_name)
            cv2.imshow(win_name, im_resize)
            cv2.moveWindow(win_name, 10, 10)

            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            elif k == ord('c'):
                cv2.destroyWindow(win_name)

        # save error
        save_path = wd+'error_analyse'

    avg_precise = avg_precise/len(img_names)
    avg_recall = avg_recall/len(img_names)

    tot_fp = tot_fp_cls + tot_fp_bb + tot_fp_redun

    print("avg_precise:%f  avg_recall:%f"%(avg_precise,avg_recall))
    print("tot_pt_bb:%d  tot_gt_bb:%d"%(tot_pt_bb_num,tot_gt_bb_num))
    print("fn:%d; fp:%d; fp_cls:%d; fp_bb:%d; fp_redun:%d;"%(tot_fn,tot_fp,tot_fp_cls,tot_fp_bb,tot_fp_redun))

if __name__ == "__main__":

    # {(model_name,dataset_name),...,...}
    # 模型文件(.weight)前缀
    ds_prefix = 'yolo-voc-800_'

    # 本次实验的名称(同一个模型可以用来做不同类型的实验)
    ds_test = 'missfresh-yolo-voc-800'

    # perform object detection on these dataSets
    sets = ['val']

    # checkpoints of models
    checkpoints = [15000]

    DataSets = make_dataset(prefix=ds_prefix, test_info=ds_test, sets=sets, iterations=checkpoints)

    # get predict results
    for ds in DataSets:
        model_name = ds[0]
        dataset_name = ds[1]
        test_info = ds[3]

        if dataset_name=='test':
            #main_plot_bb_on_mix(dataset_name, model_name)
            pass
        else:
            main_error_analyse(dataset_name, model_name,test_info,conf_thres=0.2,should_show=False)