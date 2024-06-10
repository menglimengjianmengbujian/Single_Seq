# 一 安装相关依赖，按照requirements.txt来
# 二 运行prepare文件夹下的脚本
这个实验数据是保存在两个npy文件中，x为(1330, 1, 224, 224, 3)保存的是图像文件，y为(1330,)保存的是标签文件，标签文件中1表示良性，0表示恶性
运行prepare.py文件夹下的脚本,将所有的文件按照7:1:2保存到train、valid、test三个文件夹中,可以保存到dataset文件夹中
prepare.ipynb的prepare文件步骤更加详细

    │───train
    │───valid           
    │───test          
   
# 三 要训练就运行train_fusion(concat).py
    修改好argparse中的参数
    在models文件夹下，有多个模型可以选择
# 四 要测试就运行test(concat).py
    修改好保存的模型路径和测试集的路径 
# 五 要画Grad_CAM图就运行Grad_CAM.py
    修改好文件里的路径，就可以画出每张图片的原图和Grad_CAM图了
# 五 对结果进行可视化就运行drawing.ipynb
    可以画loss图，边训练便查看loss的收敛情况；及准确率图
    可以画ROC图，查看模型的分类效果
    可以画PR曲线，查看模型的分类效果
    可以画混淆矩阵，查看模型的分类效果


