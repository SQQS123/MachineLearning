# -*-coding:utf-8-*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 导入K近邻分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 使用加载器读取数据并且存入变量iris
iris = load_iris()

# 查验数据
# print(iris.data.shape)

# 查看数据说明
# print(iris.DESCR)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 使用K近邻分类器对测试数据进行类别预测，预测结果储存在变量y_predict中
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

# 评估
# 使用模型自带的评估函数进行准确性测评
print('The accuracy of K-Nearest Neighbor Classifier is', knc.score(X_test, y_test))
# 使用sklearn.metrics里面的classification_report
print(classification_report(y_test, y_predict, target_names=iris.target_names))
