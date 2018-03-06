import numpy as np
import os
import glob

import pdb

def make_testset():
    img_dir = '/home/tfl/workspace/dataSet/GoodsID/images-raw/mix/'
    img_path_list = glob.glob(img_dir+'*.jpg')

    img_name_list = [x.split('/')[-1].split('.')[0] for x in img_path_list]

    save_path = '/home/tfl/workspace/dataSet/GoodsID/ImageSets/Main/test.txt'

    f = open(save_path,'w')
    for name in img_name_list:
        f.write(name+'\n')
    f.close()

    #pdb.set_trace()

if __name__ == "__main__":
    make_testset()