# importing required liabraries for machine learning algorithums
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import  accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from pandas.plotting import scatter_matrix

import pandas
import matplotlib.pyplot as plt

# reading 'Human Activity Recognition' dataset.
dataset_columns = ['x1','y1','z1','x2','y2','z2','x3','y3','z3','class']
har_dataset = pandas.read_csv('HAR_Dataset.csv',usecols=dataset_columns)

# splitting the dataset into trainng and test dataset
splitarray = har_dataset.values
X = splitarray[:, 0:9]
Y = splitarray[:, 9]
testdata_size = 0.20
fix_split = 7
X_trainingdataset,X_testdataset,Y_trainingdataset,Y_testdataset = model_selection.train_test_split(X,Y,test_size=testdata_size,
                                                                                                              random_state=fix_split )

# Algorithms
algorithms = []
algorithms.append(('KNN',KNeighborsClassifier()))
algorithms.append(('CART',DecisionTreeClassifier()))
algorithms.append(('RF',RandomForestClassifier()))

# Applying algorithms one by one on the dataset
algorithm_results = []
names = []
accuracy = 'accuracy'
for name,algorithm in algorithms:
    crossfold_dataset = model_selection.KFold(n_splits=10,random_state=fix_split)
    crossfold_dataset_results = model_selection.cross_val_score(algorithm,X_trainingdataset,Y_trainingdataset,cv=crossfold_dataset,scoring=accuracy)
    algorithm_results.append(crossfold_dataset_results)
    names.append(name)
    algorithm_accuracy = "%s: %f (%f)" % (name, crossfold_dataset_results.mean(),crossfold_dataset_results.std())
    print(algorithm_accuracy)

# Graph for algorithm comparison
algo_graph = plt.figure()
algo_graph.suptitle('Algorithm Comparison')
subplot = algo_graph.add_subplot(111)
plt.boxplot(algorithm_results)
subplot.set_xticklabels(names)
plt.show()

# predictions on test dataset
kNeighbourClassifier = KNeighborsClassifier()
kNeighbourClassifier.fit(X_trainingdataset,Y_trainingdataset)
predictions = kNeighbourClassifier.predict(X_testdataset)
print(accuracy_score(Y_testdataset,predictions))
print(classification_report(Y_testdataset,predictions))