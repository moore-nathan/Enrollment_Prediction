#Logistic Regression v2
#final project for Machine Learning II

#Dummy variable trap idea

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from CSV_Format import csv_format

desired_width = 100
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',100)

df = pd.read_csv("Test_AS_data_3.csv")

# lists to depict which categories are which
cat_names = ["State", "Country", "Ethnic Code", "Denomination", "Intended Major 1"]  # category names
cont_names = ['SAT_COMP', 'ACT_COMP', 'Expected Financial Contribution', 'STANDING', 'HS GPA']
YN = ['Enrolled', 'Home State of PA', 'Home County of Cambria', 'Gender', 'Admitted', 'Housing Type',
          'Legacy', 'Roster Athlete']

# call to reformat the data frame
df = csv_format(df, cat_names, cont_names, YN)

#create dummy variables for categorical data
cat_vars = df[cat_names]
cat_list = pd.get_dummies(cat_vars, drop_first=True)
final = df.join(cat_list)
final = final.drop(cat_names, axis="columns")
# print(final.head())
# data=df[[i for i in log if i not in cat_vars]]
# y=df[[‘y’]]
# x=df[[i for i in log if i not in y]]

#Made csv to make sure it actually removed and converted everything correctly
# final.to_csv("test.csv", index=False)

X = final.drop("Enrolled", axis="columns")
y = final.Enrolled
# print(X)
# print(y)

percent = .2
seed = 1

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=percent, random_state=seed, shuffle=True)


# shape of the dataset
print('Shape of training data :',train_x.shape, train_y.shape)
print('Shape of testing data :',test_x.shape, test_y.shape)


# import statsmodels.api as sm
# logit_model=sm.Logit(y,X)
# result=logit_model.fit()
# print(result.summary2())



LogModel = LogisticRegression()
SVCModel =SVC()
KNN = KNeighborsClassifier(n_neighbors=100)
KM = KMeans()
forest = RandomForestClassifier()
GBoost = GradientBoostingClassifier()
XGBC = XGBClassifier()

def stats(model):
    model.fit(train_x, train_y)
    # coefficeints of the trained model
    # print('Coefficient of model :', model.coef_)

    # intercept of the model
    # print('Intercept of model',model.intercept_)

    # predict the target on the train dataset
    # predict_train = model.predict(train_x)
    # print('Target on train data',predict_train)

    # Accuray Score on train dataset
    # accuracy_train = accuracy_score(train_y,predict_train)
    # print('accuracy_score on train dataset : ', accuracy_train)

    # predict the target on the test dataset
    predict_train = model.predict(train_x)
    predict_test = model.predict(test_x)
    # print('Target on test data',predict_test)

    # Accuracy Score on test dataset
    # accuracy_test = accuracy_score(test_y,predict_test)
    # print('accuracy_score on test dataset : ', accuracy_test)

    #Confusion Matrix
    print(confusion_matrix(test_y, predict_test))
    print("Train Accuracy:", metrics.accuracy_score(train_y, predict_train))
    print("Accuracy:",metrics.accuracy_score(test_y, predict_test))
    print("Precision:",metrics.precision_score(test_y, predict_test))
    print("Recall:",metrics.recall_score(test_y, predict_test))

    # prob = model.predict_proba(test_x)
    # p = pd.DataFrame(prob, columns=model.classes_)
    # print(p)


print('Logistic Regression')
stats(LogModel)
# stats(SVCModel)
print("K Nearest Neighbors")
stats(KNN)
# stats(KM)
print("Random Forest")
stats(forest)
print("Gradient Boosting")
stats(GBoost)
print('XGBC')
stats(XGBC)
