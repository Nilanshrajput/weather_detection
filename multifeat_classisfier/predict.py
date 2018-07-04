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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


from sklearn.metrics import classification_report, accuracy_score
import glob
import pandas as pd


df_lbp1 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\csv_kmeantrials\dfcloudy.csv')
df_lbp2 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\csv_kmeantrials\dffoggy.csv')
df_lbp3 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\csv_kmeantrials\dfrainy.csv')
df_lbp4 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\csv_kmeantrials\dfsnowy.csv')
df_lbp5 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\csv_kmeantrials\dfsunny.csv')

df_hog1 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\dfcloudy_hog.csv')
df_hog2 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\dffoggy_hog.csv')
df_hog3 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\csv_hog\dfrainyhog.csv')
df_hog4 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\dfsnowyhog.csv')
df_hog5 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\dfsunny_hog.csv')


df_sift1 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\Bow_sift-opencv-2.4.11(env-opencv)\dfcloudy.csv')
df_sift2 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\Bow_sift-opencv-2.4.11(env-opencv)\dffoggy.csv')
df_sift3 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\Bow_sift-opencv-2.4.11(env-opencv)\dfrainy.csv')
df_sift4 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\Bow_sift-opencv-2.4.11(env-opencv)\dfsnowy.csv')
df_sift5 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\Bow_sift-opencv-2.4.11(env-opencv)\dfsunny.csv')

df_clrgr1 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\dfcloudy.csv').iloc[0:, 1:64]
df_clrgr2 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\dffoggy.csv').iloc[0:, 1:64]
df_clrgr3 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\dfrainy.csv').iloc[0:, 1:64]
df_clrgr4 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\dfsnowy.csv').iloc[0:, 1:64]
df_clrgr5 = pd.read_csv('C:\Users\Abhi\dev\weather_detectcnn\dfsunny.csv').iloc[0:, 1:64]


df1 = pd.concat([df_clrgr1.iloc[0:, 1:], df_sift1.iloc[0:, 1:], df_hog1.iloc[0:, 1:], df_lbp1.iloc[0:, 1:]], axis=1, sort=False, ignore_index=True)
print "111111111111111111111111111111111111111111111111111"
print df1
df1 = df1.dropna()
print df1
a, b = df1.shape
label1 = np.ones([a, 1]) * 0
df1.to_csv('C:\Users\Abhi\dev\weather_detectcnn\multifeat_classisfier\dfcloudy.csv')


df2 = pd.concat([df_clrgr2.iloc[0:, 1:], df_sift2.iloc[0:, 1:], df_hog2.iloc[0:, 1:], df_lbp2.iloc[0:, 1:]], axis=1, sort=False, ignore_index=True)
print "222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222"
print df2
df2 = df2.dropna()
print df2
a, b = df2.shape
label2 = np.ones([a, 1]) * 1
df2.to_csv('C:\Users\Abhi\dev\weather_detectcnn\multifeat_classisfier\dffoggy.csv')


df3 = pd.concat([df_clrgr3.iloc[0:, 1:], df_sift3.iloc[0:, 1:], df_hog3.iloc[0:, 1:], df_lbp3.iloc[0:, 1:]], axis=1, sort=False, ignore_index=True)
print "333333333333333333333333333333333333333333333333333333333"
print df3
df3 = df3.dropna()
print df3
a, b = df3.shape
label3 = np.ones([a, 1]) * 2
df3.to_csv('C:\Users\Abhi\dev\weather_detectcnn\multifeat_classisfier\dfrainy.csv')


df4 = pd.concat([df_clrgr4.iloc[0:, 1:], df_sift4.iloc[0:, 1:], df_hog4.iloc[0:, 1:], df_lbp4.iloc[0:, 1:]], axis=1, sort=False, ignore_index=True)
print "444444444444444444444444444444444444444444444444444444444444444444444444444"
print df4
df4 = df4.dropna()
print df4

a, b = df4.shape
label4 = np.ones([a, 1]) * 3
df4.to_csv('C:\Users\Abhi\dev\weather_detectcnn\multifeat_classisfier\dfsnowy.csv')


df5 = pd.concat([df_clrgr5.iloc[0:, 1:], df_sift5.iloc[0:, 1:], df_hog5.iloc[0:, 1:], df_lbp5.iloc[0:, 1:]], axis=1, sort=False, ignore_index=True)

print "555555555555555555555555555555555555555555555555555555555555555555555555555"
print df5
df5 = df5.dropna()
a, b = df5.shape
label5 = np.ones([a, 1]) * 4
df5.to_csv('C:\Users\Abhi\dev\weather_detectcnn\multifeat_classisfier\dfsunny.csv')


df = pd.DataFrame(df1)
label = pd.DataFrame(label1)


df = pd.concat([df, df2], axis=0, ignore_index=True)
label = pd.concat([label, pd.DataFrame(label2)], axis=0, ignore_index=True)


df = pd.concat([df, df3], axis=0, ignore_index=True, sort=False)
label = pd.concat([label, pd.DataFrame(label3)], axis=0, ignore_index=True)
print df

df = pd.concat([df, df4], axis=0, ignore_index=True, sort=False)
label = pd.concat([label, pd.DataFrame(label4)], axis=0, ignore_index=True)
print df

df = pd.concat([df, df5], axis=0, ignore_index=True, sort=False)
label = pd.concat([label, pd.DataFrame(label5)], axis=0, ignore_index=True)
df = df.dropna(axis='columns')
print df
print df.shape
print "lllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll"
print label
print df.shape

# label.to_csv('C:\Users\Abhi\dev\weather_detectcnn\Bow_sift-opencv-2.4.11(env-opencv)\df500_label.csv')

# df.to_csv('C:\Users\Abhi\dev\weather_detectcnn\Bow_sift-opencv-2.4.11(env-opencv)\df500_siftbow.csv')


trainData = df.values
responses = label.values.ravel()

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

X_train, X_test, y_train, y_test = train_test_split(trainData, responses, test_size=0.2, random_state=40)
#clf = svm.SVC()
#clf = LinearSVC(random_state=0)
'''
clf = GridSearchCV(SVC(), tuned_parameters, cv=5)

clf.fit(X_train, y_train)
print(clf.best_params_)
print np.sort(y_test)
print y_test.sort
print y_test
y_pred = clf.predict(X_test)
print y_pred


print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

#scores = cross_val_score(clf, X_train, y_train, cv=5)
#print (scores)
'''


param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
rf.fit(X_train, y_train)
print(rf.best_params_)
pred = rf.predict(X_test)
print("Accuracy: " + str(accuracy_score(y_test, pred)))
print('\n')
print(classification_report(y_test, pred))

#scores = cross_val_score(rf, X_train, y_train, cv=5)
#print (scores)


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print("Accuracy: " + str(accuracy_score(y_test, pred)))
print('\n')
print(classification_report(y_test, pred))

scores = cross_val_score(rf, X_train, y_train, cv=5)
print (scores)
