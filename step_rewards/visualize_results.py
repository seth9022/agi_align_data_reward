import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




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

    column_names = ["episode", "paperclips", "pollution", "reward"]
   
    for column in b.columns:
        if column == '0':
            pass
        else:
            plt.plot(b[column], label='b')
            plt.plot(pb[column], label='pb')
            plt.plot(mb[column], label='mb')
            plt.title(column_names[int(column)])
            plt.legend()
            plt.show()



            plt.plot(i[column], label='i')
            plt.plot(pi[column], label='pi')
            plt.plot(mi[column], label='mi')
            plt.title(column_names[int(column)])
            plt.legend()
            plt.show()


def plot_morality(indexs):
    dfs = []
    path = "data/simulation_"
    for i in indexs:
        sim_path = f"{path}{i}/episodic_data.csv"
        df = pd.read_csv(sim_path)
        dfs.append(df)

    avg = avg_df(indexs)
    column_names = ["","Paperclips Produced", "Pollution"]
    
    titles = ["","Paperclip Production","Pollution"]
    
    df1, df2, df3= dfs[0], dfs[1], dfs[2]
   
    for column in df1.columns:
        if column == '0':
            pass
        else:
            title = f"{titles[int(column)]} with starting inventory"
            plt.plot(df1[column], label='Maximise Paperclips')
            plt.plot(df2[column], label='Full Pollution')
            plt.plot(df3[column], label='Scaled Pollution')
            plt.xlabel("Episode")
            plt.ylabel(column_names[int(column)])
            plt.title(title)
            plt.xlim(0)
            plt.ylim(0)
            plt.legend(title="Reward Function")
            plt.show()


"""def scatter_morality(indexs):
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
    plt.show()"""


plot_morality([0,2,4])
plot_morality([1,3,5])



"""
3=incontext
4=conscious
5=apathetic
7=maximise
"""