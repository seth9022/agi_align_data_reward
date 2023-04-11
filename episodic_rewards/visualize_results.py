import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


b = [0,6,11]
i = [1,7,12]
pb = [2,8,13]
pi = [3,9,14]
mb = [4,10,15]
mi = [5,16,17]
mn = [18]


def avg_df(sim_indexs):
    dfs = []
    path = "data/simulation_"
    for i in sim_indexs:
        sim_path = f"{path}{i}/episodic_data.csv"
        df = pd.read_csv(sim_path)
        dfs.append(df)

    df_avg = pd.concat(dfs).groupby(level=0).mean()
    return df_avg


def get_averages(b,i,pb,pi,mb,mi):
    b = avg_df(b)
    i = avg_df(i)
    pb = avg_df(pb)
    pi = avg_df(pi)
    mb = avg_df(mb)
    mi = avg_df(mi)

    titles = ["","Paperclip Production","Pollution", ""]
    column_names = ["episode", "paperclips", "pollution", "reward"]
   
    for column in b.columns:
        if column == '0' or column == '3':
            pass
        else:
            title = f"{titles[int(column)]} with no starting inventory"
            plt.plot(b[column], label='Maximise Paperclips')
            plt.plot(pb[column], label='Full Pollution')
            plt.plot(mb[column], label='Sentiment Pollution')
            plt.xlabel("Episode")
            plt.ylabel(column_names[int(column)])
            plt.title(title)
            plt.xlim(0)
            #plt.ylim(0)
            plt.legend(title="Reward Function")
            plt.show()



            title = f"{titles[int(column)]} with starting inventory"
            plt.plot(i[column], label='Maximise Paperclips')
            plt.plot(pi[column], label='Full Pollution')
            plt.plot(mi[column], label='Sentiment Pollution')
            plt.xlabel("Episode")
            plt.ylabel(column_names[int(column)])
            plt.title(title)
            plt.xlim(0)
            #plt.ylim(0)
            plt.legend(title="Reward Function")
            plt.show()




def scatter_morality(indexs):
    dfs = []
    path = "data/simulation_"
    for i in indexs:
        sim_path = f"{path}{i}/episodic_data.csv"
        df = pd.read_csv(sim_path)
        dfs.append(df)
    
    df = pd.concat(dfs)

    colourmap = {'somewhat desirable' : 'g', 'very desirable' : 'b', 'somewhat undesirable' : 'y', 'very undesirable' : "r"}
    colors = df['4'].map(colourmap)

    df.plot.scatter(x='1',
                    y='2',
                    c=colors)
    
    clf = linear_model.LogisticRegression()
    df_desirable = df
    df_desirable= df_desirable.replace('very desirable', 1)
    df_desirable = df_desirable.replace('somewhat desirable', 0)
    df_desirable= df_desirable.replace('somewhat undesirable', 0)
    df_desirable= df_desirable.replace('very undesirable', 0)

    y = df_desirable['4'].to_numpy()
    X = df_desirable.drop(['0','3','4'], axis=1).to_numpy()

    clf.fit(X=X,y=y)
  


    plt.scatter([], [], c='b', label='Very Desirable', marker='o')
    plt.scatter([], [], c='g', label='Somewhat Desirable', marker='o')
    plt.scatter([], [], c='y', label='Somewhat Undersirable', marker='o')
    plt.scatter([], [], c='r', label='Very Undersirable', marker='o')
    

    plt.xlabel("Paperclips Produced")
    plt.ylabel("Pollutionn")
    plt.title("Sentiment distribution for morality agent with starting inventory")
    plt.legend(title="Moral Sentiment")

    #plt.plot(x, y_pred, color='black', label='Logistic Regression Curve')
        
    
    
    plt.show()



scatter_morality(mn)

#get_averages(b,i,pb,pi,mb,mi)

