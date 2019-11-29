# TODO:use logistic regression
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_mldata
from sklearn import metrics
from sklearn.metrics import accuracy_score

# download and read mnist
mnist = fetch_mldata('MNIST original', data_home='./')

# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
images = mnist.data
targets = mnist.target

# make the value of pixels from [0, 255] to [0, 1] for further process
X = mnist.data / 255.
Y = mnist.target
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

lr = LogisticRegression()   # Logistic回归模型
lr.fit(X_train, Y_train)        # 根据数据[x,y]，计算回归参数
print ('Learning of LogisticRegression is OK...')

train_accuracy = lr.score(X_train, Y_train)
test_accuracy = lr.score(X_test, Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
