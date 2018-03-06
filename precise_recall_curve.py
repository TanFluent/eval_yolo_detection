import os,sys


def main():
    # --path
    classes = ['beer', 'beverage', 'instantnoodle', 'redwine', 'snack', 'springwater', 'yogurt']

    # --paths
    wd = '/home/tfl/workspace/project/YI/goodsid/'

    model_cfg_path = os.path.join(wd, 'cfg', 'yolo-voc.cfg')
    model_weights_path = os.path.join(wd, 'models_bn_3w_lr_0.001', 'yolo-voc_final.weights')
    meta_path = os.path.join(wd, 'cfg', 'goodid.data')

    # dataSets
    dataset_dir = '/home/tfl/workspace/dataSet/GoodsID'

    dataset = 'val'
    dataset_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '%s.txt' % dataset)

    # predict results
    thres = 0.001
    predict_results_dir = os.path.join(dataset_dir, 'predict', dataset)

    # --get testing data
    fp = open(dataset_file, 'r')
    img_names = fp.readlines()
    fp.close()


    pass


if __name__ == "__main__":
    pass
