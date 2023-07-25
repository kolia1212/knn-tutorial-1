import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('breast-cancer-wisconsin.data.txt', header=None)

col_names = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion',
             'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
df.columns = col_names
df.drop('Id', axis=1, inplace=True)

df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'],
                                  errors='coerce')  # Convert data type of Bare_Nuclei to integer because df.info() shows that Bare_Nuclei is 'object' and not 'int64'

# Data Visualization
plt.rcParams['figure.figsize'] = (30, 25)
df.plot(kind='hist', bins=10, subplots=True, layout=(5, 2), sharex=False,
        sharey=False)  # all the variables are positively skewed

correlation = df.corr()
plt.figure(figsize=(10, 8))
plt.title('Correlation of Attributes with Class variable')
a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)
# We see, that Class is highly positive correlated with Uniformity_Cell_Size, Uniformity_Cell_Shape and Bare_Nuclei

print('Before filling the missing data:')
print(df.isna().sum())  # Bare_Nuclei has 16 missing values
print()

BN_median = df['Bare_Nuclei'].median()
df['Bare_Nuclei'].fillna(BN_median, inplace=True)

print('After filling the missing data:')
print(df.isna().sum())
print()

print(df['Class'].value_counts() / np.float64(
    len(df)))  # Also we have balanced data. 2 stands for benign and 4 stands for malignant cancer
print()

X = df.drop(['Class'], axis=1)  # feature vector
y = df['Class']  # target variable

# Split data into separate training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)  # split X and y into training and testing sets

cols = X_train.columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])


# Fit K Neighbours Classifier to the training eet
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# Check accuracy score
y_pred = knn.predict(X_test)

print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
print()

y_pred_train = knn.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))
print()
# 0.9714 and 0.9821 are very good results. These two values are quite comparable. So, there is no question of overfitting


#Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)
print()


# Turns out, that if we set k(number of neighbours)=7, then the result will be slightly better
knn_7 = KNeighborsClassifier(n_neighbors=7)
knn_7.fit(X_train, y_train)
y_pred_7 = knn_7.predict(X_test)
print('Model accuracy score with k=7 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_7)))
print()

# Confusion matrix for k=7:
cm_7 = confusion_matrix(y_test, y_pred_7)
print('Confusion matrix for k=7:\n\n', cm_7)
print()

# plt.show()
