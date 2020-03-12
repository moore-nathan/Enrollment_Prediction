# (Class) used to look at independent variables individually vs y
import matplotlib.pyplot as plt


class independentStats:
    def __init__(self, df, cont_names):
        self.df = df
        self.cont_names = cont_names


    def x_vs_y(self,x,y):
        plt.subplot()


    @staticmethod
    def plotting(df_x, probs, cont_names):
        cont_names.remove('ACT_COMP')
        n=1
        for i in cont_names:
            plt.subplot(2,2,n)
            plt.scatter(df_x.loc[:,i],probs)
            plt.title(i)
            n+=1
        plt.show()
