import numpy as np
import os
import pdb
from utils import *
import matplotlib.pyplot as plt

from conf import *

# #############
# Functions
# #############

def main_plot_eval_results_of_models(model_infos,should_plot=True,dataset=['train','val']):
    '''
    plot curves of MAPs

    :param model_infos:
    :param should_plot:
    :param dataset:
    :return:
    '''

    eval_res_dir = os.path.join(wd,'results', 'predict')

    model_num = len(model_infos)
    dataset_num = len(dataset)

    mAPs = np.zeros((model_num,dataset_num))
    fg_bg_APs = np.zeros((model_num,dataset_num))

    subcls_APs = {}
    for idx,cls in enumerate(classes):
        subcls_APs[cls] = np.zeros((model_num,dataset_num))

    for idx,name in enumerate(model_infos):
        if not os.path.exists(os.path.join(eval_res_dir,name[1],name[0])):
            print('No such folder!')
            print name
            exit()

        model_eval_res_dir = os.path.join(eval_res_dir,name[1],name[0])
        for ds in dataset:
            if not os.path.isfile(os.path.join(model_eval_res_dir,'%s_mAP.txt'%ds)):
                print ('No such file!')
                print os.path.join(model_eval_res_dir,'%s_mAP.txt'%ds)
                exit()

        train_eval_path = os.path.join(model_eval_res_dir,'train_mAP.txt')
        val_eval_path = os.path.join(model_eval_res_dir, 'val_mAP.txt')

        # get train MAP from train_mAP.txt
        train_f = open(train_eval_path,'r')
        train_data = train_f.read().strip().split('\n')
        train_f.close()

        # get val MAP from val_mAP.txt
        val_f = open(val_eval_path, 'r')
        val_data = val_f.read().strip().split('\n')
        val_f.close()

        # train
        for line in train_data:
            k = line.split(':')[0]
            v = float(line.split(':')[1])
            if k=='mAP':
                mAPs[idx,0] = v
            elif k=='fg_bg_AP':
                fg_bg_APs[idx,0] = v
            else:
                if k in classes:
                    subcls_APs[k][idx,0] = v
                else:
                    print("error class name:%s"%k)
                    exit()
        # val
        for line in val_data:
            k = line.split(':')[0]
            v = float(line.split(':')[1])
            if k == 'mAP':
                mAPs[idx, 1] = v
            elif k == 'fg_bg_AP':
                fg_bg_APs[idx, 1] = v
            else:
                if k in classes:
                    subcls_APs[k][idx,1] = v
                else:
                    print("error class name:%s"%k)
                    exit()
    # --plot
    iters_num = [int(x[0].split('_')[1].strip()) for x in model_infos]
    xticks = range(len(iters_num))
    xticklabels = iters_num

    # -plot "all_cls_ap"/"fg_bg_ap"
    plot_list = ['mAP', 'fg_bg_AP']  # define what we should plot
    ap_dict = {}

    for item in plot_list:
        if item == 'mAP':
            data = mAPs
            title = item
        elif item == 'fg_bg_AP':
            data = fg_bg_APs
            title = item
        else:
            print("false input")
            exit()

        ap_dict[item]=data

        ax_xlabel = 'Iteration'
        ax_ylabel = 'AP'
        savename = item

        if should_plot:
            plot_ap_trends(data, title, ax_xlabel, ax_ylabel, xticks, xticklabels, savename)

    # plot multi cls ap
    ax_xlabel = 'Iteration'
    ax_ylabel = 'AP'
    title = 'Multi classes AP'
    if should_plot:
        plot_multi_cls_ap_trends(subcls_APs, title, ax_xlabel, ax_ylabel, xticks, xticklabels, 'Multi_classes_AP')

    return ap_dict,subcls_APs,xticks,xticklabels

def plot_ap_trends(ap_data,title,xlabel,ylabel,xticks,xticklabels,savename):
    """
    Plot ap-iteration curve

    :param ap_data:
    :param title:
    :param xlabel:
    :param ylabel:
    :param xticks:
    :param xticklabels:
    :param savename:
    :return:
    """
    _, (ax) = plt.subplots(1, 1, sharex=True)
    ax.plot(ap_data[:, 0], '-or')
    ax.plot(ap_data[:, 1], '-.ob')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    plt.savefig('%s.png'%savename)

def plot_ap_trends_diff_exp(test_infos,legend_names,ap_data_dict,title,xlabel,ylabel,xticks,xticklabels,savename):
    ''' Plot AP trends of diff experiment results

    :param prefix:
    :param ap_data_dict: {'prefix':ap_data}
    :param title:
    :param xlabel:
    :param ylabel:
    :param xticks:
    :param xticklabels:
    :param savename:
    :return:
    '''
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']*10
    _, (ax) = plt.subplots(1, 1, sharex=True)

    cnt = 0

    for idx,item in enumerate(test_infos):
        #pdb.set_trace()
        ap_dict = ap_data_dict[item]

        ax.plot(ap_dict['mAP'][:, 0], color=colors[cnt], label='%s %s' % (legend_names[idx], 'train'), linestyle='-',linewidth=2,marker='o')
        ax.plot(ap_dict['mAP'][:, 1], color=colors[cnt], label='%s %s' % (legend_names[idx], 'val'), linestyle='-.', linewidth=1,marker='s')

        cnt = cnt + 1

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    plt.legend(loc='upper left')
    plt.savefig('%s.png'%savename)

def plot_multi_cls_ap_trends(ap_data_dict,title,xlabel,ylabel,xticks,xticklabels,savename):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']*10

    cnt = 0
    _, (ax) = plt.subplots(1, 1, sharex=True)
    for k,v in ap_data_dict.items():

        ax.plot(v[:, 0], color=colors[cnt],label=k,linestyle='-',linewidth=2)
        ax.plot(v[:, 1], color=colors[cnt],label=k,linestyle='-.',linewidth=1)

        cnt = cnt + 1

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    plt.legend(loc='upper left')
    #plt.show()
    plt.savefig('%s.png'%savename)


# #####################
# Main
# #####################
if __name__ == "__main__":

    SingleExp = False

    ######################################
    # single experiment results

    if SingleExp:
        # prefix of yolo-model file (.weight)
        ds_prefix = 'yolo-voc-800_'

        # folder of predicting results (default dir: ./results/predict/)
        ds_test = 'missfresh-yolo-voc-800'

        # checkpoints of models
        checkpoints = [2000,4000,15000]

        # {(model_name,dataset_name),...,...}
        modelSets = make_dataset_model_only(prefix=ds_prefix,test_info=ds_test,iterations=checkpoints)

        # plot evaluation results
        main_plot_eval_results_of_models(modelSets)

        exit()

    ######################################
    # diff experiment compare

    # prefix of yolo-models
    prefixs = ['yolo-voc-800_']
    # folders of predict results
    test_infos = ['missfresh-yolo-voc-800']
    # names of axis legend
    legend_names = ['missfresh-yolo-voc-800']

    # checkpoints of models
    checkpoints = [2000, 4000, 15000]

    tot_ap_dict = {}
    tot_cls_ap_dict = {}
    tot_xticks_dict = {}
    tot_xticklabels_dict = {}

    for idx,item in enumerate(test_infos):
        #
        ap_dict = {}
        cls_ap_dict = {}
        xticks = ''
        xticklabels = ''
        #
        modelsets = make_dataset_model_only(prefix=prefixs[idx],test_info=item,iterations=checkpoints)
        #
        ap_dict,cls_ap_dict,xticks,xticklabels = main_plot_eval_results_of_models(modelsets,should_plot=False)

        tot_ap_dict[item] = ap_dict
        tot_cls_ap_dict[item] = cls_ap_dict
        tot_xticks_dict[item] = xticks
        tot_xticklabels_dict[item] = xticklabels


    plot_ap_trends_diff_exp(test_infos,legend_names, tot_ap_dict,'mAPs of diff exp', 'iters', 'mAP', xticks, xticklabels, 'mAP_of_diff_exp')