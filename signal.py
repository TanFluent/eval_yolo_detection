#coding:utf-8 允许中文注释
import xml.etree.ElementTree as ET
import xml.etree as et
import os
import time
from os import listdir, getcwd
from os.path import join

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import matplotlib.patches as patches

import random
import numpy as np
import cv2

#p3、p4分别为groundtruth框的左上和右下角点，p1、p2为检测框的
def Overlap_Rate(p1x,p1y,p2x,p2y,p3x,p3y,p4x,p4y):
    over_rate=0
    over_p1x=0
    over_p1y=0
    over_p2x=0
    over_p2y=0
    area_gt=(p4y-p3y)*(p4x-p3x)#groundtruth框的面积
    area_test=(p2y-p1y)*(p2x-p1x)#检测框的面积
    if((p2y>p3y)and(p1y<p4y)and(p2x>p3x)and(p1x<p4x)):#如果有重合
        over_p1x=max(p1x,p3x)#重叠区左上角点
        over_p1y=max(p1y,p3y)
        over_p2x=min(p2x,p4x)#重叠区右下角点
        over_p2y=min(p2y,p4y)
        area_over=(over_p2y-over_p1y)*(over_p2x-over_p1x)
        over_rate=area_over/(area_gt+area_test-area_over)#重合区面积比两框并集的面积
#        print("U并区面积：",area_gt)
#        print("重合区域面积：",area_over)
    return over_rate,over_p1x,over_p1y,over_p2x,over_p2y


foldername_test='test0125'
foldername_gt='gt0125'

filepath_test='E:\\笔记本桌面9.29\\陌上花Yi+\\货架商品识别\\Annotation\\'+foldername_test+'\\'
filepath_gt='E:\\笔记本桌面9.29\\陌上花Yi+\\货架商品识别\\Annotation\\'+foldername_gt+'\\'

list_test = os.listdir(filepath_test)
list_gt = os.listdir(filepath_gt)

#用于存test和groundtruth各自文件夹下的xml文件名
namelist_test=[]
namelist_gt=[]
signal_gt=[] #groundtruth框的标识位，标识为0表示没被用过，为1表示已被用过
for i in list_test:
    files_test=os.path.splitext(i)[0]#os.path.splitext(i)[1]就是后缀如.jpg
    namelist_test.append(files_test)
for i in list_gt:
    files_gt=os.path.splitext(i)[0]
    namelist_gt.append(files_gt)

#标识位初始化为0，假设一个xml文件里面最多只有999个矩形框
xml_num=999
for j in range(0,xml_num):
    signal_gt.append(0)
    
    
count=0 #总的xml文件的个数
test_num=0 #每个xml文件中检测框的个数
for i in range(0,len(namelist_test)):
    count=count+1
    files_test_use=str(namelist_test[i])
    files_gt_use=str(namelist_gt[i])
    
    in_file_test = open('E:\\笔记本桌面9.29\\陌上花Yi+\\货架商品识别\\Annotation\\'+foldername_test+'\\'+files_test_use+'.xml','r', encoding='UTF-8')
    tree_test=ET.parse(in_file_test)
    root_test = tree_test.getroot()
    in_file_gt = open('E:\\笔记本桌面9.29\\陌上花Yi+\\货架商品识别\\Annotation\\'+foldername_gt+'\\'+files_gt_use+'.xml','r', encoding='UTF-8')
    tree_gt=ET.parse(in_file_gt)
    root_gt = tree_gt.getroot()
    
    image_path='D:\\Text_image\\Product_Retrieval\\01hongjiu\\frames_00558.jpg'
    image= cv2.imread(image_path)
    
    num_test=0
    num_gt=0
    num_test_true=0
    #遇到文本中无法读取的 root_test[i][1] 自行中断
    for i in range(6,9999):
        num_test=num_test+1
        try:  
            str1=root_test[i][1].text  
        except Exception as err:  
#            print(err)  
            break
#    print('num_test:',count,num_test)
    for i in range(6,9999):
        num_gt=num_gt+1
        try:  
            str1=root_gt[i][1].text  
        except Exception as err:  
#            print(err)  
            break
#    print('num:',count,num_test)
    
    for i in range(6,6+num_test-1):
        xmin_test = int(root_test[i][4][0].text)
        ymin_test = int(root_test[i][4][1].text)
        xmax_test = int(root_test[i][4][2].text)
        ymax_test = int(root_test[i][4][3].text)
        cv2.rectangle(image,(xmin_test-6,ymin_test-6),(xmax_test+6,ymax_test+6),(255,255,255),2,8,0);
        
        for j in range(6,6+num_gt-1):
            xmin_gt = int(root_gt[j][4][0].text)
            ymin_gt = int(root_gt[j][4][1].text)
            xmax_gt = int(root_gt[j][4][2].text)
            ymax_gt = int(root_gt[j][4][3].text)
            cv2.rectangle(image,(xmin_gt,ymin_gt),(xmax_gt,ymax_gt),(0,0,255),2,8,0);
            
            over_rate,over_p1x,over_p1y,over_p2x,over_p2y= \
            Overlap_Rate(xmin_test,ymin_test,xmax_test,ymax_test, xmin_gt,ymin_gt,xmax_gt,ymax_gt)
            
            if(over_rate>0.3):# 大于0.3表示检测框预测到了正确的groundtruth
                if(signal_gt[j]==0):
                    num_test_true=num_test_true+1
                    signal_gt[j]=1
                    cv2.rectangle(image,(xmin_test,ymin_test),(xmax_test,ymax_test),(255,0,0),1,8,0);
                    cv2.rectangle(image,(xmin_test+3,ymin_test+3),(xmax_test-3,ymax_test-3),(0,255,0),3,8,0);
                    cv2.rectangle(image,(xmin_test-2,ymin_test-2),(xmax_test+2,ymax_test+2),(0,0,255),1,8,0);
#    cv2.imshow("Source Iamge",image)
#    cv2.waitKey(0)               
            
    print('groundtruth的框数目：',num_gt)
    print('test框的数目：',num_test)
    print('检测出True框的数目：',num_test_true)
    print('准确率：','%.2f'%(100*num_test_true/num_test),'%')
    print('招回率：','%.2f'%(100*num_test_true/num_gt),'%')
    time_str=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    print ('System Time:',time_str)
    
    str_count=str(count)
    cv2.imwrite('D:\\Text_image\\Product_Retrieval\\result\\_result\\result_'+str_count+'.jpg',image)
    
    #读完一个xml文件之后把标识位清零
    for j in range(0,xml_num):
        signal_gt[j]=0






