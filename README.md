# Plate-Recognition 车牌识别

## 概要
这个程序试图使用OCR技术识别车牌, 主要步骤如下:
1. 从包含一张车牌的图片中提取出车牌(待完成)
2. 将得到的车牌图像进行切割, 降噪等预处理
3. 将处理后的车牌图片切分成单个字符(1个汉字+6个数字/字母)
4. 对单个字符进行识别

## 车牌提取
待完成

## 车牌预处理
1. 切割可能存在的边框
2. 降噪(待完成)

## 字符切分
1. DFS寻找连通块
2. 选取最大的7个联通块作为待识别字符
3. 最左边的连通块(汉字块)单独再处理(待完善)

## 字符识别
使用基本的机器学习方法.
1. 预先训练出系数矩阵的偏置矩阵
2. 将每个字符标准化
3. 得到识别结果

## 待完成项目
1. 车牌提取
2. 车牌降噪

## 待完善项目
1. 汉字切割