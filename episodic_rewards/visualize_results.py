import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn import linear_model


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
        if column == '0' or column == '3' or column == '4'  or column == '5':
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



def plot_100_data():
    path = "data/simulation_2/episodic_data.csv"
    df = pd.read_csv(path)
     
    titles = ["","Paperclip Production In Encoded Environment","Pollution In Encoded Environment", ""]
    column_names = ["episode", "paperclips", "pollution", "reward"]
   
    for column in df.columns:
        if column == '1' or column == '2':
    
            title = f"{titles[int(column)]}"
            plt.plot(df[column], label='Encoded Agent')
          #  plt.plot(functional_reward_df[column], label='Functional Reward')
            plt.xlabel("Episode")
            plt.ylabel(column_names[int(column)])
            plt.title(title)
            plt.xlim(0)
            #plt.ylim(0)
            plt.legend(title="Agent")
            plt.show()
    scatter_morality(path, "Incontext Agent in encoded environemnt")

def plot_1000_data():


    human_path = "data/human/episodic_data.csv"
    human_df = pd.read_csv(human_path)

    apathetic_path = "data/apathetic/episodic_data.csv"
    apathetic_df = pd.read_csv(apathetic_path)

    environmental_path = "data/enviromental/episodic_data.csv"
    environmental_df = pd.read_csv(environmental_path)

    max_path = "data/max/episodic_data.csv"
    max_df = pd.read_csv(max_path)

    functional_reward_path = "data/functional_reward_starting_inv/simulation_1/episodic_data.csv"
    functional_reward_df = pd.read_csv(functional_reward_path)

    
    titles = ["","Paperclip-focused Agent Paperclip Production","Paperclip-focused Agent Pollution", ""]
    column_names = ["episode", "paperclips", "pollution", "reward"]
   
    for column in human_df.columns:
        if column == '0' or column == '3' or column == '4'  or column == '5':
            pass
        else:
            title = f"{titles[int(column)]}"
            #plt.plot(apathetic_df[column], label='Apathetic Agent', c='r')
            #plt.plot(environmental_df[column], label='Concious Agent', c='g')
            plt.plot(human_df[column], label='Agent with incontext prompting')
            plt.plot(max_df[column], label='Max Production Agent')
            plt.plot(functional_reward_df[column], label='Functional Reward')
            plt.xlabel("Episode")
            plt.ylabel(column_names[int(column)])
            plt.title(title)
            plt.xlim(0)
            plt.ylim(1500)
            plt.legend(title="Agent", loc=0)
            plt.show()

def scatter_morality(path, title):
    df = pd.read_csv(path)
       
    """
   # lf = linear_model.LogisticRegression()
    df_desirable = df
    df_desirable= df_desirable.replace('very desirable', 1)
    df_desirable = df_desirable.replace('somewhat desirable', 0)
    df_desirable= df_desirable.replace('somewhat undesirable', 0)
    df_desirable= df_desirable.replace('very undesirable', 0)

    y = df_desirable['4'].to_numpy()
    X = df_desirable.drop(['0','3','4'], axis=1).to_numpy()

    clf.fit(X=X,y=y)
  """


    colourmap = {'somewhat desirable' : 'g', 'very desirable' : 'b', 'somewhat undesirable' : 'y', 'very undesirable' : "r"}
    colors = df['6'].map(colourmap)

    df.plot.scatter(x='1',
                    y='2',
                    c=colors,
                    s=10)

    plt.scatter([], [], c='b', label='Very Desirable', marker='o')
    plt.scatter([], [], c='g', label='Somewhat Desirable', marker='o')
    plt.scatter([], [], c='y', label='Somewhat Undersirable', marker='o')
    plt.scatter([], [], c='r', label='Very Undersirable', marker='o')
    

    plt.xlabel("Paperclips Produced")
    plt.ylabel("Pollution")
    plt.title(f"Sentiment distribution for {title}")
    plt.legend(title="Moral Sentiment")

    #plt.plot(x, y_pred, color='black', label='Logistic Regression Curve')
        
    
    
    plt.show()


plot_100_data()

import sys; sys.exit(1)
human_path = "data/human/episodic_data.csv"
human_df = pd.read_csv(human_path)

apathetic_path = "data/apathetic/episodic_data.csv"
apathetic_df = pd.read_csv(apathetic_path)

environmental_path = "data/enviromental/episodic_data.csv"
environmental_df = pd.read_csv(environmental_path)

max_path = "data/max/episodic_data.csv"
max_df = pd.read_csv(max_path)

plot_1000_data()
scatter_morality(human_path, "Agent with Incontext Prompts")
scatter_morality(apathetic_path, "Agent with Apathetic view on Environment")
scatter_morality(environmental_path, "Agent with Concious view on Environment")
scatter_morality(max_path, "Agent Maximizing Paperclip Production")

#get_averages(b,i,pb,pi,mb,mi)

