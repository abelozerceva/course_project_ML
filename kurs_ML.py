import numpy as np
import random
import cv2
import time
import matplotlib.pyplot  as plt
from graphviz import Source
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from matplotlib.colors import ListedColormap


def accuracy(clf):
    cv_scores = cross_val_score(estimator=clf, X=X_test, y=y_test, cv=5)
    return cv_scores.mean()

# зададим параметры набора данных
img_w = 1280
img_h = 720
aspect_ratio_0 = 3 # h/w
aspect_ratio_1 = 0.5
w_max_0 = img_w/8
w_max_1 = img_w/4
scale = 1/img_h
N = 200 #число примеров на класс

#сгенерируем данные
data_clean = np.zeros((2*N, 8))

#установим следующий порядок: cls, w, h, bl_x, bl_y, w_m, h_m, k
# print(data_clean.shape)
for i in range(2*N):
    if i < N:
        cls = 0
        w = w_max_0*np.random.rand(1)
        h = w*aspect_ratio_0
    else:
        cls = 1
        w = w_max_1*np.random.rand(1)
        h = w*aspect_ratio_1
    bl_y = (img_h-h)*np.random.rand(1) #положение нижней левой точки
    bl_x = (img_w-w)*np.random.rand(1)
    k = bl_y*scale
    w_m = w*k #метрическая ширина в зависимости от положения
    h_m = h*k
    data_clean[i,:] =  cls, w, h, bl_x, bl_y, w_m, h_m, k

np.set_printoptions(threshold = 10, edgeitems = 10, formatter={'float': '{: 0.3f}'.format})
# print(data_clean)

img_blank = np.zeros((img_h,img_w, 3))
img_blank[:] = (255, 255, 255)
color_dict = {0: (0,0,255), 1: (30,205,30), 2: (255,0,0)} 
for id, r in enumerate(data_clean[0:400:17]):
    #print(r[3]+r[1])
    cv2.rectangle(img_blank, (int(r[3]), int(r[4]+r[2])),
                  (int(r[3]+r[1]), int(r[4])), color_dict[r[0]], 2)
plt.figure(figsize=(12, 12))
# print("img_blank",img_blank.shape)
img_blank = img_blank.astype(np.uint8)
plt.imshow(img_blank,cmap='gray')
plt.show()

data_noise = data_clean.copy()
# print((data_noise[:,1]*(np.random.rand(2*N)-0.5))/2)
data_noise[:, 1] = data_noise[:,1]+(data_noise[:,1]*(np.random.rand(2*N)-0.5))/2
data_noise[:, 2] = data_noise[:,2]+(data_noise[:,2]*(np.random.rand(2*N)-0.5))/2
data_noise[:, 5] = data_noise[:,5]+(data_noise[:,5]*(np.random.rand(2*N)-0.5))/2
data_noise[:, 6] = data_noise[:,6]+(data_noise[:,6]*(np.random.rand(2*N)-0.5))/2

img_blank = np.zeros((img_h,img_w, 3))
img_blank[:] = (255, 255, 255)
color_dict = {0: (0,0,255), 1: (30,205,30), 2: (255,0,0)} 
for id, r in enumerate(data_noise[0:400:17]):
    #print(r[3]+r[1])
    cv2.rectangle(img_blank, (int(r[3]), int(r[4]+r[2])),(int(r[3]+r[1]), int(r[4])), color_dict[r[0]], 2)
plt.figure(figsize=(12, 12))
# print("img_blank",img_blank.shape)
img_blank = img_blank.astype(np.uint8)
plt.imshow(img_blank,cmap='gray')
plt.show()

# print(data_noise)

data_bad = np.zeros((N, 8))
# print(data_bad.shape)
for i in range(N):
    cls = 2
    w = (img_w/6)*np.random.rand(1)
    h = w*aspect_ratio_1*np.random.uniform(1.0,2.0,1)
    bl_y = (img_h-h)*np.random.rand(1) #положение нижней левой точки
    bl_x = (img_w-w)*np.random.rand(1)
    k = bl_y*scale*np.random.rand(1)
    w_m = w*k #метрическая ширина в зависимости от положения
    h_m = h*k
    data_bad[i,:] =  cls, w, h, bl_x, bl_y, w_m, h_m, k

for id, r in enumerate(data_bad[0:200:17]):
    cv2.rectangle(img_blank, (int(r[3]), int(r[4]+r[2])),(int(r[3]+r[1]), int(r[4])), color_dict[r[0]], 2)
plt.figure(figsize=(12, 12))
print("img_blank",img_blank.shape)
img_blank = img_blank.astype(np.uint8)
plt.imshow(img_blank,cmap='gray')
plt.show()

X_data = data_noise[:,1:]
Y_data = data_noise[:,0]

# KNeighborsClassifier
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data , test_size=0.7)
knn_clf = KNeighborsClassifier(n_neighbors=10, metric = "minkowski")
knn_clf.fit(X_train, y_train)
KNN_acc = accuracy(knn_clf)
print('Accuracy kNN', KNN_acc)
end_time = time.time()
knn_time = end_time-start_time
print("Time for 5 fold CV on KNN is:", knn_time)

knn_probs = knn_clf.predict_proba(X_test)
n_uncertain_probs = 0
for prob in knn_probs:
    if (prob[0]>=0.4) and (prob[0]>=0.6):
        n_uncertain_probs = n_uncertain_probs +1
print("количество примеров", knn_probs.shape[0])
print("количество неуверенных классификаций ", n_uncertain_probs)
print("доля неуверенных классификаций", n_uncertain_probs/knn_probs.shape[0])

# DecisionTreeClassifier
start_time = time.time()
tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_leaf = 5)
tree_clf.fit(X_train, y_train)
tree_acc =  accuracy(tree_clf)
print('Accuracy decision trees', tree_acc)
end_time = time.time()
tree_time = end_time-start_time
print("Time for 5 fold CV on decision trees is:", tree_time)

crit = 'entropy'
dotfile = export_graphviz(tree_clf, feature_names=[ "w", "h", "bl_x", "bl_y", "w_m", "h_m", "k"], class_names=["person", "car"], out_file=None, filled=True, node_ids=True)
graph = Source(dotfile)
# Сохраним дерево как toy_example_tree_X.png, где Х - entropy или gini, критерий качестве рабиения
graph.format = 'png'
graph.render("tree1_example_tree_{}".format(crit),view=True)
tree1 = cv2.imread("tree1_example_tree_entropy.png")
plt.figure(figsize=(15, 15))
plt.imshow(tree1,cmap='gray')
plt.show()

# Упростим данные
X_data = data_simple[:,1:]
Y_data = data_simple[:,0]
# print(X_data.shape)
# print(Y_data.shape)
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data , test_size=0.7)

start_time = time.time()
tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_leaf = 5)
tree_clf.fit(X_train, y_train)
tree_acc =  accuracy(tree_clf)
print('Accuracy decision trees', tree_acc)
end_time = time.time()
tree_time = end_time-start_time
print("Time for 5 fold CV on decision trees is:", tree_time)

data_simple = np.zeros((2*N, 4))
data_simple[:,0] = data_noise[:,0]
data_simple[:,1] = data_noise[:,2]/data_noise[:,1]
data_simple[:,2] = data_noise[:,4]
data_simple[:,3] = data_noise[:,7]
# print(data_simple)
crit = 'entropy'
dotfile = export_graphviz(tree_clf, feature_names=['aspect_ratio', 'bl', "k"], class_names=["person", "car"], out_file=None, filled=True, node_ids=True)
graph = Source(dotfile)
# Сохраним дерево как toy_example_tree_X.png, где Х - entropy или gini, критерий качестве рабиения
graph.format = 'png'
graph.render("tree2_example_tree_{}".format(crit),view=True)
tree2 = cv2.imread("tree2_example_tree_entropy.png")
plt.figure(figsize=(15, 15))
plt.imshow(tree2,cmap='gray')
plt.show()

start_time = time.time()
bayes_clf = GaussianNB()
bayes_clf.fit(X_train, y_train)
bayes_acc = accuracy(bayes_clf)
print('Accuracy bayes', bayes_acc)
end_time = time.time()
bayes_time = end_time-start_time
print("Time for 5 fold CV on bayessss:P is:", bayes_time)

# Трехклассовая классификация
data_combined = np.zeros((N*3, 8))
data_combined[:2*N,:] = data_noise[:,:]
data_combined[2*N:,:] = data_bad[:,:]
# print(data_combined.shape)
X_data = data_combined[:,1:]
Y_data = data_combined[:,0]
# print(X_data.shape)
# print(Y_data.shape)
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data , test_size=0.7)

# KNeighborsClassifier
start_time = time.time()
knn_clf = KNeighborsClassifier(n_neighbors=10, metric = "minkowski")
knn_clf.fit(X_train, y_train)
KNN_acc = accuracy(knn_clf)
print('Accuracy kNN', KNN_acc)
end_time = time.time()
knn_time = end_time-start_time
print("Time for 5 fold CV on KNN is:", knn_time)

# DecisionTreeClassifier
start_time = time.time()
tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_leaf = 5)
tree_clf.fit(X_train, y_train)
tree_acc =  accuracy(tree_clf)
print('Accuracy decision trees', tree_acc)
end_time = time.time()
tree_time = end_time-start_time
print("Time for 5 fold CV on decision trees is:", tree_time)

# GaussianNB
start_time = time.time()
bayes_clf = GaussianNB()
bayes_clf.fit(X_train, y_train)
bayes_acc = accuracy(bayes_clf)
print('Accuracy bayes', bayes_acc)
end_time = time.time()
bayes_time = end_time-start_time
print("Time for 5 fold CV on bayessss:P is:", bayes_time)