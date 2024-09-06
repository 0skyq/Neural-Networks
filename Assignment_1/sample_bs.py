import csv
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np

# read csv file into list and convert elements to int
with open('balance-scale.csv', 'r') as read_obj:
	csv_reader = csv.reader(read_obj)
	str_data = list(csv_reader)
	for i in range(len(str_data)):
		if str_data[i][0]=='B':
			str_data[i][0] = 0
		elif str_data[i][0]=='L':
			str_data[i][0] = 1
		elif str_data[i][0]=='R':
			str_data[i][0] = 2
	dataset = [list(map(int, i)) for i in str_data]

# Separate labels and features for each data item
# print(dataset[:5])
X, y = [row[1:] for row in dataset], [row[0] for row in dataset]

# create 5 fold cross validation set (n_splits=5)
rn = range(1, len(dataset))
kf5 = KFold(n_splits=5, shuffle=True, random_state=4)

# traverse over each split
for train_index, test_index in kf5.split(rn):
	# train and test splits
	# print(train_index, test_index)
	X_train, X_test = np.array([X[i] for i in train_index]), np.array([X[i] for i in test_index])
	y_train, y_test = np.array([y[i] for i in train_index]), np.array([y[i] for i in test_index])
	print(X_train)
	print(y_train)
	
	# Save the datasets to file to process it from file separately
	
	# create decision tree using the sklearn library
	clf = DecisionTreeClassifier()
	#print(clf)
	# more at: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
	y_pred = clf.fit(X_train, y_train).predict(X_test)
	print(y_pred)
	print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['B', 'L', 'R']))
