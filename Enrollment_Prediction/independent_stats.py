# (Class) used to look at independent variables individually vs y
import pandas
import matplotlib.pyplot as plt
from sklearn import metrics
from collections import Counter


class independentStats:
    def __init__(self, df, cont_names):
        self.df = df
        self.cont_names = cont_names

    @staticmethod
    def bar_graph(df, cat_names):
        N = df[df['Enrolled'] == 0]
        Y = df[df['Enrolled'] == 1]

        for i in cat_names:
            # print(Y[i].value_counts(sort=False))
            dickY = dict((Y[i].value_counts(sort=False)))
            dickN = dict((N[i].value_counts(sort=False)))
            for j in df[i].unique():
                if (j in dickY) and (j in dickN):
                    dickY[j] = (dickY[j], dickN[j])
                elif (j in dickY) and (j not in dickN):
                    dickY[j] = (dickY[j], 0)
                else:
                    try:
                        dickY[j] = (0, dickN[j])
                    except:
                        print("bruh")
            # print(dickY)
            Y1 = [k[0] for k in dickY.values()]
            N1 = [k[1] for k in dickY.values()]
            fuck = pandas.DataFrame({"yes": Y1, "no": N1}, index=dickY.keys())
            fuck.plot.bar(logy=True)
            plt.title(i)
            plt.show()
        # [df[i].value_counts().plot(kind='bar') for i in cat_names]
        # df['Ethnic Code'].value_counts().plot(kind='bar')

    @staticmethod
    def x_vs_y(x,y):
        plt.subplot()

    @staticmethod
    def PRC(test_y, predict_test):
        # calculate precision-recall curve
        precision, recall, thresholds = metrics.precision_recall_curve(test_y, predict_test)
        #plot the precision-recall curve
        no_skill = len(test_y[test_y==1]) / len(test_y)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label='Model')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
        # print(model.evals_result())

    @staticmethod
    def plotting(df_x, probs, cont_names):
        cont_names.remove('ACT_COMP')
        n=1
        for i in cont_names:
            plt.subplot(2,3,n)
            plt.scatter(df_x.loc[:,i],probs)
            plt.title(i)
            n+=1
        plt.show()

    @staticmethod
    def ROC_AUC(model, test_x, test_y):
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(test_y))]
        # predict probabilities
        lr_probs = model.predict_proba(test_x)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # calculate scores
        ns_auc = metrics.roc_auc_score(test_y, ns_probs)
        lr_auc = metrics.roc_auc_score(test_y, lr_probs)
        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Model: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        ns_fpr, ns_tpr, _ = metrics.roc_curve(test_y, ns_probs)
        lr_fpr, lr_tpr, _ = metrics.roc_curve(test_y, lr_probs)
        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Model')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()


    @staticmethod
    def stats(model, train_x, train_y, test_x, test_y):
        model.fit(train_x, train_y)
        # model.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], eval_metric='rmse')

        # predict the target on the test dataset
        predict_train = model.predict(train_x)
        predict_test = model.predict(test_x)

        print("Confusion Matrix:",metrics.confusion_matrix(test_y, predict_test))
        print("Train Accuracy:", metrics.accuracy_score(train_y, predict_train))
        print("Accuracy:",metrics.accuracy_score(test_y, predict_test))
        print("Precision:",metrics.precision_score(test_y, predict_test))
        print("Recall:",metrics.recall_score(test_y, predict_test))
        print("F1 score:", metrics.f1_score(test_y,predict_test))
        fpr, tpr, thresholds = metrics.roc_curve(test_y, predict_test)
        print('AUC: %.3f' % metrics.roc_auc_score(test_y,predict_test))



