# -*-coding:utf-8-*-
# 导入手写体数字加载器
from sklearn.datasets import load_digits
# model_selection用于数据分割
from sklearn.model_selection import train_test_split
# 导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 导入基于线性假设的支持向量机分类器LinearSVC
from sklearn.svm import LinearSVC
# 使用下面的模块对预测结果做更加详细的分析
from sklearn.metrics import classification_report

# 检视数据规模和特征维度
# 这里的digits应该就是一些手写体数字的图片
digits = load_digits()

# print(digits.data.shape)

# 分成1:3两份
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)

# 检视训练与测试数据规模
# print(y_train.shape)
# print(y_test.shape)

# 将特征数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 初始化线性假设的支持向量机分类器LinearSVC
lsvc = LinearSVC()
# 进行模型训练
lsvc.fit(X_train, y_train)
# 利用训练良好的模型对测试样本的数字类别进行预测，预测结果储存在变量y_predict中
y_predict = lsvc.predict(X_test)
# print(y_predict)

# 结果评估
# print('The Accuracy of Linear SVC is ',lsvc.score(X_test,y_test))

# 使用sklearn.metrics里的classification_report这个模块进行详细的分析
print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))
