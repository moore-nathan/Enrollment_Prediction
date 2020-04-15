# Classification models
# final project for Machine Learning II

import random as rnd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from Enrollment_Prediction.CSV_Format import csv_format
import Enrollment_Prediction.pre_processing as pre_processing
from Enrollment_Prediction.independent_stats import independentStats as IS
from tkinter import *
from tkinter import filedialog
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.cluster import KMeans
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# GUI for grabbing a file
def get_file():
    root = Tk()
    root.title("Enrollment Prediction")
    root.iconbitmap('Images/SFU.ico')

    e = Entry(root, width=50)
    label = Label(text="Select a file")
    def open():
        root.filename = filedialog.askopenfilename(initialdir='C:\\Users\\SFU\\PycharmProjects\\Enrollment_Prediction\\Data', title="Select CSV", filetypes=(("CSV", "*.csv"),("All files","*.*")))
        e.insert(0, root.filename)

    def submit():
        root.filename = e.get()
        root.destroy()

    button_select = Button(root, text="...", command=open, padx=10)
    button_submit = Button(root, text="Submit", command=submit)
    label.grid(row=0, column=0, columnspan=2)
    e.grid(row=1, column=0)
    button_select.grid(row=1, column=1)
    button_submit.grid(row=3)
    root.mainloop()
    return root.filename

# feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    # chi-square and mutual info classifier produce identical results
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def main():
    desired_width = 100
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns',100)

    cols = ["Student ID", "State", "Country", "Ethnic Code", "Denomination", "Intended Major 1 Dept", 'SAT_COMP', 'ACT_COMP',
            'Expected Financial Contribution', 'STANDING', 'HS GPA', 'Enrolled', 'Home State of PA', 'Home County of Cambria',
            'Gender', 'Admitted', 'Housing Type', 'Legacy', 'Athlete', 'MARKET_SEG', 'Brooks Rating', 'Honors Program Interest']
    # part 2 will include 'MARKET_SEG', 'Brooks Rating', 'Honors Program Interest'

    # switching intended major with intended department

    ######################################################################

    # filename = get_file()
    filename = 'Data/Admitted.csv'
    #######################################################################
    # df = pd.read_csv("Admitted.csv", low_memory=False)  # low_memory turns off auto dtype checking (maybe)
    df = pd.read_csv(filename, low_memory=False)
    df = df[cols]

    # lists to depict which categories are which
    cat_names = ["State", "Country", "Ethnic Code", "Denomination", "Intended Major 1 Dept"]  # category names
    cont_names = ['SAT_COMP', 'ACT_COMP', 'Expected Financial Contribution', 'STANDING', 'HS GPA',
                  'Brooks Rating']
    YN = ['Enrolled', 'Home State of PA', 'Home County of Cambria', 'Gender', 'Admitted', 'Housing Type',
              'Legacy', 'Athlete', 'Honors Program Interest', 'MARKET_SEG']

    # call to reformat the data frame
    df = csv_format(df, cat_names, cont_names, YN)

    # call to process the data
    df = pre_processing.pre_process(df)

    #create dummy variables for categorical data
    cat_vars = df[cat_names]
    cat_list = pd.get_dummies(cat_vars, drop_first=True)
    final = df.join(cat_list)
    final = final.drop(cat_names, axis="columns")
    #removing the ID's from final
    # tested and for XGBoost everything must be int/float/bool

    #Splitting up X(independent variables) and y(dependent variable)
    X = final.drop("Enrolled", axis="columns")
    y = final.Enrolled

    # # test cases for testing no dummy variables
    # X = df.drop("Enrolled", axis="columns")
    # y = df.Enrolled

    percent = .33
    seed = rnd.randint(1,1000)

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=percent, random_state=seed, shuffle=True)

    #grabbing the IDS of just the test data and then dropping that column (this will only be done in testing)
    # print(test_x)
    ID = list(test_x.loc[:, 'Student ID'])

    train_x = train_x.drop("Student ID", axis="columns")
    test_x = test_x.drop("Student ID", axis="columns")

    # import statsmodels.api as sm
    # logit_model=sm.Logit(y,X)
    # result=logit_model.fit()
    # print(result.summary2())

    # LogModel = LogisticRegression()
    # SVCModel =SVC()
    # KNN = KNeighborsClassifier(n_neighbors=100)
    # KM = KMeans()
    # forest = RandomForestClassifier()
    # GBoost = GradientBoostingClassifier()
    XGBC = XGBClassifier()


    # models= {"Logistic Regression": LogModel, "K Nearest Neighbors": KNN, "Random Forest": forest, "Gradient Boosting":GBoost, "XGBC": XGBC}

    print("XGBC testing")
    IS.stats(XGBC, train_x, train_y, test_x, test_y)
    XGBC.fit(test_x, test_y)

    # print(XGBC.feature_importances_)
    # print(X.columns)

    # X_train_fs, X_test_fs, fs = select_features(train_x, train_y, test_x)

    for name, feature in zip(X.columns, XGBC.feature_importances_):
        print("%s: importance:%.3f" %(name,feature))

    # plt.bar(range(len(XGBC.feature_importances_)), XGBC.feature_importances_)
    # plt.show()

    prob = XGBC.predict_proba(test_x)
    # print(prob)
    # probs = pd.DataFrame({'ID': ID, 'Not_Enrolling_Percentage': prob[:, 0], 'Enrolling_Percentage': prob[:, 1]},
    #                      columns=['ID', 'Not_Enrolling_Percentage', 'Enrolling_Percentage'])
    # probs.to_csv("C:\Users\SFU\PycharmProjects\Enrollment_Prediction\Data\Prob_test.csv", index=False)
    # IS.plotting(test_x, prob[:, 1], cont_names)

    IS.bar_graph(df, cat_names)
