# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

colun_names = ['样例编号', '肿块厚度', '细胞大小均匀性', '细胞形状均匀性', '边缘粘附', '单层上皮细胞大小', '裸核', '染色质', '正常核仁', '有丝分裂', '种类']
data = pd.read_csv(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
    names=colun_names)

# 使用标准缺失值来替换掉问号,所以说，，为啥会有问号
data = data.replace(to_replace='?', value=np.nan)

data = data.dropna(how='any')

# print(data)
# 以上步骤得到乳腺癌肿瘤数据

# 使用下面的函数将我们得到的乳腺癌数据分为两部分
# 75%的训练集:X_train->算法->y_train
# 25%的测试集:X_test->算法->y_test
X_train, X_test, y_train, y_test = train_test_split(data[colun_names[1:10]], data[colun_names[10]], test_size=0.25,
                                                    random_state=33)
# print(y_train.value_counts())
# print(y_test.value_counts())

# 以上步骤得到分割的数据结果

# 标准化数据，保证每个维度的特征数据方差为1，均值为0.使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)





# 初始化LogisticRegression与SGDClassifier,两种预测模型
lr = LogisticRegression()
sgdc = SGDClassifier()

# LR(逻辑斯蒂)训练
# 调用LogisticRegression中的fit函数/模块用来训练模型参数
lr.fit(X_train, y_train)

# 使用得到训练得到的预测算法lr.predict()对X_test预测，预测结果存储在lr_y_predict中
# 使用训练良好的模型lr对X_test进行预测，结果存储在变量lr_y_predict中
lr_y_predict = lr.predict(X_test)
print("线性回归预测结果：")
print(lr_y_predict)





# SGDC(随机梯度下降)训练
sgdc.fit(X_train, y_train)
# 使用训练好的模型sgdc对X_test进行预测，结果储存在变量sgdc_y_predict中
sgdc_y_predict = sgdc.predict(X_test)
print("随机梯度下降预测结果:")
print(sgdc_y_predict)





# 以上步骤得到了预测结果

# 使用逻辑斯蒂回归模型自带的评分函数score获得模型在测试集上的准确性结果
print('Accuracy of LR Classifier:',lr.score(X_test,y_test))
print(classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant']))

# 使用随机梯度下降模型自带的评分函数score获得模型在测试集上的准确性结果
print('Accuracy of SGDC Classifier:', sgdc.score(X_test, y_test))
# 利用classification_report模块获得SGDClassfier其他三个指标的结果
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))
