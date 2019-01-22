# -*-coding:utf-8-*-
import pandas as pd
from sklearn.model_selection import train_test_split
# 使用scikit-learnn.feature_extraction中的特征转换器,详见3.1.1.1特征抽取
from sklearn.feature_extraction import DictVectorizer
# 从sklearn.tree中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
# 随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# 梯度提升决策树
from sklearn.ensemble import GradientBoostingClassifier
# 分析性能
from sklearn.metrics import classification_report

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# print(titanic.head())
# print(titanic.info())

# 机器学习有一个不太被初学者重视并且耗时，但是十分重要的一环——特征的选择，这个需要基于一些背景知识。根据我们对这场事故的了解,sex,age,pclass这些特征都很有可能是决定幸免与否的关键因素
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# print(X.info())

# 借由上面的输出，我们设计如下几个数据处理的任务
# 首先我们补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略
X['age'].fillna(X['age'].mean(), inplace=True)
# 由下面的输出得知age特征值得到了补完
# print(X.info())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

vec = DictVectorizer(sparse=False)

X_train = vec.fit_transform(X_train.to_dict(orient='record'))
# print(vec.feature_names_)

# 对测试数据的特征进行转换
X_test = vec.transform(X_test.to_dict(orient='record'))

# 使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()
# 使用分割到的训练数据进行模型学习
dtc.fit(X_train, y_train)
# 用训练好的决策树模型对测试特征数据进行预测
dtc_y_predict = dtc.predict(X_test)

# 使用决策树模型对泰坦尼克好乘客是否生还性能预测
print(dtc.score(X_test, y_test))
# 输出更详细的分类性能
print(classification_report(dtc_y_predict, y_test, target_names=['died', 'survived']))

#
#
##########################################################################################
# 使用集成模型预测(需要上面的单一决策树)
#

# 使用随机森林分类器进行集成模型的训练以及预测分析
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)

# 使用梯度提升决策树进行集成模型的训练以及预测分析
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)

# 集成模型对泰坦尼克号乘客是否生还的预测性能
print('The acuracy of decision tree is', rfc.score(X_test, y_test))
print(classification_report(dtc_y_predict, y_test))

# 输出随机森林分类器在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标
print('The accuracy of random forest classifier is', dtc.score(X_test, y_test))
print(classification_report(rfc_y_pred, y_test))

# 输出梯度提升决策树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标
print('The accuracy of gradient tree boosting is', gbc.score(X_test, y_test))
print(classification_report(gbc_y_pred, y_test))
