import numpy as np
import os
import pdb
from utils import *
import matplotlib.pyplot as plt

# working dir
wd = '/home/tfl/workspace/project/YI/goodsid/'
# dataSets dir
dataset_dir = '/home/tfl/workspace/dataSet/GoodsID'
# classes
classes = ['beer','beverage','instantnoodle','redwine','snack','springwater','yogurt']


def main_plot_eval_results_of_models(model_infos,should_plot=True):
    #model_names = [x[0] for x in model_infos]
    #test_info = [x[1] for x in model_infos]

    eval_res_dir = os.path.join(dataset_dir,'predict')
    dataset = ['train','val']

    model_num = len(model_infos)
    dataset_num = len(dataset)

    mAPs = np.zeros((model_num,dataset_num))
    fg_bg_APs = np.zeros((model_num,dataset_num))
    beer_APs = np.zeros((model_num,dataset_num))
    beverage_APs = np.zeros((model_num,dataset_num))
    instantnoodle_APs = np.zeros((model_num,dataset_num))
    redwine_APs = np.zeros((model_num,dataset_num))
    snack_APs = np.zeros((model_num,dataset_num))
    springwater_APs = np.zeros((model_num,dataset_num))
    yogurt_APs = np.zeros((model_num,dataset_num))

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

        #
        train_f = open(train_eval_path,'r')
        train_data = train_f.read().strip().split('\n')
        train_f.close()

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
            elif k=='beer':
                beer_APs[idx,0] = v
            elif k=='beverage':
                beverage_APs[idx,0] = v
            elif k=='instantnoodle':
                instantnoodle_APs[idx,0] = v
            elif k=='redwine':
                redwine_APs[idx,0] = v
            elif k=='snack':
                snack_APs[idx,0] = v
            elif k=='springwater':
                springwater_APs[idx,0] = v
            elif k=='yogurt':
                yogurt_APs[idx,0] = v
        # val
        for line in val_data:
            k = line.split(':')[0]
            v = float(line.split(':')[1])
            if k == 'mAP':
                mAPs[idx, 1] = v
            elif k == 'fg_bg_AP':
                fg_bg_APs[idx, 1] = v
            elif k == 'beer':
                beer_APs[idx, 1] = v
            elif k == 'beverage':
                beverage_APs[idx, 1] = v
            elif k == 'instantnoodle':
                instantnoodle_APs[idx, 1] = v
            elif k == 'redwine':
                redwine_APs[idx, 1] = v
            elif k == 'snack':
                snack_APs[idx, 1] = v
            elif k == 'springwater':
                springwater_APs[idx, 1] = v
            elif k == 'yogurt':
                yogurt_APs[idx, 1] = v
    # --plot
    iters_num = [int(x[0].split('_')[1].strip()) for x in model_infos]
    xticks = range(len(iters_num))
    xticklabels = iters_num

    #plot_list = ['mAP','fg_bg_AP']+classes
    plot_list = ['mAP', 'fg_bg_AP']
    ap_dict = {}

    for item in plot_list:
        if item == 'mAP':
            data = mAPs
            title = item
        elif item == 'fg_bg_AP':
            data = fg_bg_APs
            title = item
        elif item == 'beer':
            data = beer_APs
            title = 'AP of %s'%item
        elif item == 'beverage':
            data = beverage_APs
            title = 'AP of %s' % item
        elif item == 'instantnoodle':
            data = instantnoodle_APs
            title = 'AP of %s' % item
        elif item == 'redwine':
            data = redwine_APs
            title = 'AP of %s' % item
        elif item == 'snack':
            data = snack_APs
            title = 'AP of %s' % item
        elif item == 'springwater':
            data = springwater_APs
            title = 'AP of %s' % item
        elif item == 'yogurt':
            data = yogurt_APs
            title = 'AP of %s' % item

        ap_dict[item]=data

        ax_xlabel = 'Iteration'
        ax_ylabel = 'AP'
        savename = item

        if should_plot:
            plot_ap_trends(data, title, ax_xlabel, ax_ylabel, xticks, xticklabels, savename)

    # plot multi cls ap
    cls_ap_dict = {}
    for item in classes:
        if item == 'beer':
            data = beer_APs
        elif item == 'beverage':
            data = beverage_APs
        elif item == 'instantnoodle':
            data = instantnoodle_APs
        elif item == 'redwine':
            data = redwine_APs
        elif item == 'snack':
            data = snack_APs
        elif item == 'springwater':
            data = springwater_APs
        elif item == 'yogurt':
            data = yogurt_APs

        cls_ap_dict[item] = data

    ax_xlabel = 'Iteration'
    ax_ylabel = 'AP'
    title = 'Multi classes AP'
    if should_plot:
        plot_multi_cls_ap_trends(cls_ap_dict, title, ax_xlabel, ax_ylabel, xticks, xticklabels, 'Multi_classes_AP')

    return ap_dict,cls_ap_dict,xticks,xticklabels

# #####################
def plot_ap_trends(ap_data,title,xlabel,ylabel,xticks,xticklabels,savename):
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
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
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
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

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
    '''
    # {(model_name,dataset_name),...,...}
    #modelSets = [('yolo-voc_20000'), ('yolo-voc_40000')]
    modelSets = make_dataset_model_only(prefix='yolo-voc-800_',test_info='nl-yolo-voc-800')

    # plot evaluation results
    main_plot_eval_results_of_models(modelSets)


    exit()
    '''
    ######################################
    # diff experiment compare
    prefixs = ['yolo-voc-800_','yolo-voc-608_',
               'yolo-voc_','yolo-voc-800-multiscale_',
               'yolo-voc-608_']
    test_infos = ['nl-yolo-voc-800','yolo-voc-608',
                  'yolo-voc-test608','nl-yolo-voc-800-multiscale',
                  'nl-yolo-voc-608']
    legend_names = ['nl-yolo_imagenet_800','yolo_imagenet_608',
                    'yolo_imagenet_test608','yolo-imagenet-800-multiscale',
                    'nl-yolo_imagenet_608']

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
        modelsets = make_dataset_model_only(prefix=prefixs[idx],test_info=item,iterations=[10000,36000])
        #
        ap_dict,cls_ap_dict,xticks,xticklabels = main_plot_eval_results_of_models(modelsets,should_plot=False)

        tot_ap_dict[item] = ap_dict
        tot_cls_ap_dict[item] = cls_ap_dict
        tot_xticks_dict[item] = xticks
        tot_xticklabels_dict[item] = xticklabels


    plot_ap_trends_diff_exp(test_infos,legend_names, tot_ap_dict,'mAPs of diff exp', 'iters', 'mAP', xticks, xticklabels, 'mAP_of_diff_exp')