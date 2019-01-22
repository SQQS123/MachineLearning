# -*-coding:utf-8-*-
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 从读取房价数据存储在变量boston中
boston = load_boston()
# 输出数据描述
# print(boston.DESCR)

X = boston.data
y = boston.target

# 随机采样25%的数据构建测试样本，其余作为训练样本
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# 分析回归目标值的差异
print('The max target value is', np.max(boston.target))
print('The min target value is ', np.min(boston.target))
print('The average target value is ', np.mean(boston.target))

# 分别初始化对特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
# 这里书上没有写reshape(-1,1)网上搜答案无果，尝试着根据错误信息在后面加了reshape(-1,1)成功编译,但是会有两个warning
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 使用现行回归模型LinearRegression和SGDRessor分别对美国波士顿地区房价进行预测
# 使用默认配置初始化现行回归器LinearRegression
lr = LinearRegression()
# 使用训练数据进行参数轨迹
lr.fit(X_train, y_train)
# 对测试数据进行回归预测
lr_y_predict = lr.predict(X_test)

# 使用默认配置初始化线性回归器SGDRegressor
sgdr = SGDRegressor()
# 使用训练数据进行参数估计。
sgdr.fit(X_train, y_train)
# 对测试数据进行回归预测
sgdr_y_predict = sgdr.predict(X_test)

# 使用三种回归评价机制以及两种调用R-squared评价模块，并输出评价结果
print('The value of default measurement of LinearRegression is', lr.score(X_test, y_test))

# 使用r2_score模块，并输出评估结果
print('The value of R-squared of LinearRegression is', r2_score(y_test, lr_y_predict))

# 使用mean_squared_error模块，并输出评估结果
print('The mean squared error of LinearRegression is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))

# 使用mean_absolute_error模块，并输出评估结果
print('The mean absolute error of LinearRegression is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))

print('---------------------------------------分界---------------------------------------')
# 使用SGDRegression模型自带的评估模块，并输出评估结果
print('The value of default measurement of SGDRegression is', sgdr.score(X_test, y_test))

# 使用r2_score模块，并输出评估结果
print('The value of R-squared of SGDRegressor is', r2_score(y_test, sgdr_y_predict))

# 使用mean_squared_error模块，并输出评估结果
print('The mean squared error of SGDRegressor is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))

# 使用mean_absolute_error模块，并输出评估结果
print('The mean absolute error of SGDRegressor is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))
