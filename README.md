###########################
文件结构
###########################

ROOT
|       |libdarknet                     # 存放darknet.so文件
|       |       libdarknet.so
|       |material                       # 存放实验相关的素材
|       |       |cfg
|       |       |       *.data
|       |       |       *.names
|       |       |       *.cfg
|       |       |yolo_models            # 存放yolo模型文件
|       |results                        # 存放实验结果
|       |       |predict
bb_info_statistic.py                    # 统计实验数据 Bounding-Boxes(BB)的信息(未完成)
conf.py                                 # 实验相关配置文件
darknet.py                              # darknet框架的python接口
dataset_info_statistic.py               # 统计实验数据的信息
error_analyse.py                        # 错误分析，错误样本统计
evaluation.py                           # 计算map，precise，recall的核心代码，当做API使用
Get_Evaluating_Results.ipynb            # 实验结果分析流程(jupyter版，不建议使用)
plot_bb_on_img.py                       # 将Label和预测结果的BB画到对应的图片上
plot_evaluation_results.py              # 画出 bg-fg,cls,all-cls 的MAP 图
Plot_Training_Info.ipynb                # 画出训练相关的信息(jupyter版，不建议使用)
plot_training_info.py                   # 画出训练相关的信息
precise_recall_curve.py                 # 画出precise-recall曲线(未完成)
run_get_map.py                          # 获取实验结果(bg-fg,cls,all-cls MAP)
run_testing.py                          # 获取每张图片的检测结果
utils.py                                # 工具类

###########################
物体检测结果分析流程
###########################

Training
1) 画出train-loss曲线，调参，迭代训练；
    详见，plot_training_info.py

Validating/Testing
1) 在conf文件中配置本次实验的信息;
2) 运行 run_testing.py 获取每张图片物体检测的结果;
3) 运行 run_get_map.py 获取 bg-fg,cls,all-cls 的 MAP 数据;
4) 运行 plot_evaluation_results.py 将MAP数据画出来;

5) 运行 plot_bb_on_img.py 可以预览检测结果