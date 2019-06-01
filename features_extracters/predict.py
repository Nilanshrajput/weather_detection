import numpy as np
import cv2
from skimage import color
from matplotlib import pyplot as plt
from skimage.feature import hog as hg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import classification_report, accuracy_score
import Descriptors as des
import glob
import pandas as pd

df1 = pd.read_csv('dfcloudy.csv').iloc[0:, 1:64]
a, b = df1.shape
label1 = np.ones([a, 1]) * 0


df2 = pd.read_csv('dffoggy.csv').iloc[0:, 1:64]
a, b = df2.shape
label2 = np.ones([a, 1]) * 1


df3 = pd.read_csv('dfrainy.csv').iloc[0:, 1:64]
a, b = df3.shape
label3 = np.ones([a, 1]) * 2


df4 = pd.read_csv('dfsnowy.csv').iloc[0:, 1:64]
a, b = df4.shape
label4 = np.ones([a, 1]) * 3

df5 = pd.read_csv('dfsunny.csv').iloc[0:, 1:64]
a, b = df5.shape
label5 = np.ones([a, 1]) * 4


df = pd.DataFrame(df1)
label = pd.DataFrame(label1)
print label1

df = pd.concat([df, df2], axis=0, ignore_index=True)
label = pd.concat([label, pd.DataFrame(label2)], axis=0, ignore_index=True)


df = pd.concat([df, df3], axis=0, ignore_index=True, sort=False)
#label = pd.concat([label, pd.DataFrame(label3)], axis=0, ignore_index=True)
print df

df = pd.concat([df, df4], axis=0, ignore_index=True, sort=False)
label = pd.concat([label, pd.DataFrame(label4)], axis=0, ignore_index=True)
print df

df = pd.concat([df, df5], axis=0, ignore_index=True, sort=False)
label = pd.concat([label, pd.DataFrame(label5)], axis=0, ignore_index=True)
print df
print df.shape

print df.shape
label = pd.read_csv('label500_hog.csv')

df.to_csv('C:\Users\Abhi\dev\weather_detectcnn\df500_colorcogram.csv')


trainData = df.values
responses = label.values.ravel()


X_train, X_test, y_train, y_test = train_test_split(trainData, responses, test_size=0.1, random_state=4)
# clf = svm.SVC()
clf = LinearSVC(random_state=0)
clf.fit(X_train, y_train)

print y_test

y_pred = clf.predict(X_test)
print y_pred


print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

scores = cross_val_score(clf, X_train, y_train, cv=5)
print (scores)


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

scores = cross_val_score(rf, X_train, y_train, cv=5)
print (scores)
