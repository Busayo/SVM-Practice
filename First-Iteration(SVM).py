# DSN tutorial day 20
import numpy as np
from sklearn import neighbors, svm
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
# We know that there are 16 missing data as stated in the .names file, so we factor this in by making them outliers and
# our algorithm sees it as such and it has no effect on our data set.
df.replace('?', -99999, inplace=True)
# The ID column is unnecessary and as such must be dropped or it might affect our algorithm performance.
df.drop(['id'], 1, inplace=True)
# X is for features
# y is for labels
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# We're selecting a test size of 20%, we could go higher of lower. The line above shuffles the data for us for maximum
# testing without bias.
clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

# You could try dropping out the line with 'df.drop(['id'], 1, inplace=True)' and running again and you'll see the
# massive drop in accuracy.

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
# We want to try getting a prediction for made up data. From the .data file, this set of values should be benign.
example_measures = (example_measures.reshape(len(example_measures), -1))
# Instead of hard coding the reshape line above, i.e (2, -1), we just include the length of the array above.
prediction = clf.predict(example_measures)
print(prediction)
