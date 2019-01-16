# -*-coding:utf-8-*-
# 加载获得新闻的东西
from sklearn.datasets import fetch_20newsgroups
# 导入分割数据的东西
from sklearn.model_selection import train_test_split
# 从sklearn.feature_extraction.text里导入用于问恩特征向量转化模块。
from sklearn.feature_extraction.text import CountVectorizer
# 从sklearn.naive_bayes导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 从sklearn.metrics里导入classification
from sklearn.metrics import classification_report

news = fetch_20newsgroups(subset='all')

# 查看我们得到的新闻数据
print(len(news.data))
print(news.data[0])

# 分两份
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 训练+测试
# 使用默认配置初始化朴素贝叶斯模型
mnb = MultinomialNB()
# 利用训练数据对模型参数进行估计
mnb.fit(X_train, y_train)
# 预测结果保存
y_predict = mnb.predict(X_test)

# 性能评估
print('The accuracy of Naive Bayes Classifier is', mnb.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=news.target_names))
