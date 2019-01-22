# -*-coding:utf-8-*-
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR  # 导入支持向量机
from sklearn.neighbors import KNeighborsRegressor  # 导入K近邻回归器
from sklearn.tree import DecisionTreeRegressor  # 导入决策树回归器
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

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
print('---------------------------------------分界---------------------------------------')

# 分别初始化对特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
# 这里书上没有写reshape(-1,1)网上搜答案无果，尝试着根据错误信息在后面加了reshape(-1,1)成功编译
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# ###################以上除了import导入的需要多加一些，其他的都是数据的初始化#############################

# 一、使用三种不同核函数配置的支持向量机回归模型进行训练，并且分别对测试数据做出预测

# 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
# 1.linear
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

# 使用多项式核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
# 2.poly
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

# 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
# 3.rbf
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

# 使用R-squared、MSE和MAE指标对三种配置的支持向量机(回归)模型在相同测试集上进行性能评估

# linear
print('R-squared value of linear SVR is', linear_svr.score(X_test, y_test))
print('The mean squared error of linear SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print('The mean absolute error of linear SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print('---------------------------------------分界---------------------------------------')

# poly
print('R-squared value of Poly SVR is', poly_svr.score(X_test, y_test))
print('The mean squared error of Poly SVR is ',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print('The mean absolute error of Poly SVR is ',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print('---------------------------------------分界---------------------------------------')

# rbf
print('R-squared value of RBF is', rbf_svr.score(X_test, y_test))
print('The mean squared error of RBF SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print('The mean absoluate error of RBF is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print('---------------------------------------分界---------------------------------------')

# 二、使用两种不同配置的K近邻回归模型对美国波士顿房价数据进行回归预测
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

# 初始化K近邻回归器，并且调整配置，使得预测的方式为根据距离加权回归:weights='distance'。
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

# 对两种不同配置的K近邻回归模型在美国哦波士顿房价数据上进行预测性能的评估
print('R-squared value of uniform-weighted KNeighorRegression:', uni_knr.score(X_test, y_test))
print('The mean squared error of uniform-weighted KNeighorRegression:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print('The mean absoluate error of uniform-weighted KNeighorRegression:',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print('---------------------------------------分界---------------------------------------')

# 使用R-squared、MSE以及MAE三种指标对根据距离加权回归配置的K近邻模型在测试集上进行性能评估.
print('R-squared value of distance-weighted KNeighorRegression：', dis_knr.score(X_test, y_test))
print('The mean squared error of distance-weighted KNeighorRegression:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print('The mean absoluate error of distance-weighted KNeighorRegression:',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print('---------------------------------------分界---------------------------------------')

# 三、使用回归树对美国波士顿房价训练数据进行学习，并对猜测试数据进行预测

# 使用默认配置初始化决策树回归器
dtr = DecisionTreeRegressor()
# 用波士顿房价的训练数据构建回归树
dtr.fit(X_train, y_train)
# 使用默认配置的单一回归树对测试数据进行预测，并将预测值存储在变量dtr_y_predict中
dtr_y_predict = dtr.predict(X_test)

# 对单一回归树模型在美国波士顿房价测试数据上的预测性能进行评估
print('R-squared value of DeecisionTreeRegressoor:', dtr.score(X_test, y_test))
print('The mean squared error of DeecisionTreeRegressoor:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
print('The mean absoluate error of DeecisionTreeRegressoor:',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
print('---------------------------------------分界---------------------------------------')

# 四、三种集成回归模型对美国波士顿房价训练数据进行学习，并对测试数据进行预测
# 使用RandomForestRegressor训练模型，并对测试数据作出预测，结果存储在变量rfr_y_predict中
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)

# 使用ExtraTreeRegressor训练模型，并对测试数据做出预测，结果存储在变量etr_y_predict中
etr = ExtraTreesRegressor()
etr.fit(X_train, y_train)
etr_y_predict = etr.predict(X_test)

# 使用GradientBoostingRegressor训练模型，并对测试数据做出预测，结果存储在变量gbr_y_predict中
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_r_predict = gbr.predict(X_test)

# 对三种集成回归模型在美国波士顿房价测试数据上的回归性能进行评估
# 使用R-squared、MSE以及MAE指标对默认配置的随机回归森林在测试集上进行性能评估
print('R-squared value of RandoomForestRegressor：', rfr.score(X_test, y_test))
print('The mean squared error of RandomForestRegressor:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print('The mean absoluate error of RandomForestRegressor:',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print('---------------------------------------分界---------------------------------------')

# 使用R-squared、MSE以及MAE指标对默认配置的极端回归森林在测试集上进行性能评估
print('R-squared value of Exc', etr.score(X_test, y_test))
print('The mean squared error of ExtraTreeRegressor:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print('The mean absoluate error of ExtraTreeRegressor:',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print('---------------------------------------分界---------------------------------------')

# 利用训练好的极端回归森林模型，输出每种特征对预测目标的贡献度
print(np.sort(zip(etr.feature_importances_, boston.feature_names), axis=-0))
print('---------------------------------------分界---------------------------------------')

# 使用R-squared、MSE以及MAE指标对默认配置的梯队提升回归树在测试集上进行性能评估
print('R-squared value of GradientBoostingRegressor:', gbr.score(X_test, y_test))
print('The mean squared error of GradientBoostRegressor:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_r_predict)))
print('The mean absoluate error of GrandientBoostingRegressoor:',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_r_predict)))
