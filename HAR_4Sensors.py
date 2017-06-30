##############################################################################################################
# Filename    : HAR_4Sensors.py                                                                              #
# Description : This script is used for reading the recorded data of human activities from the dataset and   #
#               then further use it for data analysis. The data is from 4 sensors. The dataset is divided    #
#               into training and testing dataset. The training dataset is used for training the model and   #
#               testing dataset is used for predictions.                                                     #
##############################################################################################################

# importing required liabraries for machine learning algorithums
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import  accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.forest import RandomForestClassifier

import pandas
import matplotlib.pyplot as plt
import numpy as np

# reading 'Human Activity Recognition' dataset.
dataset_columns = ['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4','class']
har_dataset = pandas.read_csv('HAR_Dataset.csv',usecols=dataset_columns)

# splitting the dataset into trainng and test dataset
splitarray = har_dataset.values
X = splitarray[:, 0:12]
Y = splitarray[:, 12]
testdata_size = 0.20
fix_split = 7
X_trainingdataset,X_testdataset,Y_trainingdataset,Y_testdataset = model_selection.train_test_split(X,Y,test_size=testdata_size,
                                                                                                              random_state=fix_split )

# raw data graph
counts = har_dataset['class'].value_counts()
plt.bar(range(len(counts)),counts)
plt.title('Distribution of activities')
plt.ylabel('Frequency')
plt.xlabel('Activites')
activity = counts.keys()
x_pos = np.arange(len(activity))
plt.xticks(x_pos,activity)
plt.show()

# Algorithms to be used
algorithms = []
algorithms.append(('KNN',KNeighborsClassifier()))
algorithms.append(('CART',DecisionTreeClassifier()))
algorithms.append(('RF',RandomForestClassifier()))

# Applying algorithms one by one on the dataset
algorithm_results = []
names = []
accuracy = 'accuracy'
print('\n')
print('Algorithm accuracy using 4 sensors :')
print('\n')
for name,algorithm in algorithms:
    crossfold_dataset = model_selection.KFold(n_splits=10,random_state=fix_split)
    crossfold_dataset_results = model_selection.cross_val_score(algorithm,X_trainingdataset,Y_trainingdataset,cv=crossfold_dataset,scoring=accuracy)
    algorithm_results.append(crossfold_dataset_results)
    names.append(name)
    algorithm_accuracy = "%s: %f (%f)" % (name, crossfold_dataset_results.mean()*100,crossfold_dataset_results.std())
    print(algorithm_accuracy)

# Graph for algorithm comparison
algo_graph = plt.figure()
algo_graph.suptitle('Algorithm Comparison')
subplot = algo_graph.add_subplot(111)
plt.boxplot(algorithm_results)
plt.ylabel('Accuracy')
plt.xlabel('Algorithums')
subplot.set_xticklabels(names)
plt.show()

# predictions on test dataset
kNeighbourClassifier = KNeighborsClassifier()
kNeighbourClassifier.fit(X_trainingdataset,Y_trainingdataset)
predictions = kNeighbourClassifier.predict(X_testdataset)
print('\n')
print('Prediction Accuracy:',accuracy_score(Y_testdataset,predictions)*100,'%')
print('\n')
print(classification_report(Y_testdataset,predictions))