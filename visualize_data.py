import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorboard as tf
from torch.utils.tensorboard import SummaryWriter
import io

# Group the data by episode


def plot_df_1(df, tensor_writer, title):
    episodes = df.groupby('episode')

    # Iterate over episodes and create summaries
    for episode, data in episodes:
        plot_data = data.drop(columns=['action', 'episode'])
        plot_data = plot_data.set_index('step')
        plot = plot_data.plot()
    
    # Add labels and title
        plt.xlabel('Step')
        plt.ylabel('Count')
        plt.title(f'Count of {title} Over Steps for episode {episode}')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        with tensor_writer.as_default():
            tf.summary.image(f'{title} plot for episode {episode}', image, step=episode)


def plot_df(df, tensor_writer, title, item_names):
    episodes = df.groupby('episode')

    # Iterate over episodes and create summaries
    plot_steps = 10
    for episode, data in episodes:
        if int(episode) % plot_steps == 0 or int(episode) == 1:
            plot_data = data.drop(columns=['action', 'episode'])
            plot_data = plot_data.set_index('step')
            for index, row in plot_data.iterrows():
                for name in item_names:
                    tensorboard_location = "episode " + str(episode) + "/" + title 
                    tensor_writer.add_scalars(tensorboard_location, {name: row[name]}, index)



def plot_pollution(df, tensor_writer, title):
    episodes = df.groupby('episode')

    plot_steps = 10
    for episode, data in episodes:
        if int(episode) % plot_steps == 0 or int(episode) == 1:
            plot_data = data.drop(columns=['episode'])
            plot_data = plot_data.set_index('step')
            for index, row in plot_data.iterrows():
                tensorboard_location = "episode " + str(episode) + "/" + title 
                tensor_writer.add_scalars(tensorboard_location, {'pollution': row['pollution']}, index)



def plot_by_episode(path, item_names):
    crafted_df=  pd.read_csv(path + "/crafted.csv")
    inventory_df =  pd.read_csv(path + "/inventory.csv")
    effects_df =  pd.read_csv(path + "/effects.csv")
    pollution_df =  pd.read_csv(path + "/pollution.csv")


    layout = created_tensorboard_layout(crafted_df, item_names)
    summary_writer = SummaryWriter(path + "/plot_data/")
    summary_writer.add_custom_scalars(layout)

    plot_df(crafted_df, summary_writer, "crafted", item_names)
    plot_df(inventory_df, summary_writer, "inventory", item_names)
    plot_df(effects_df, summary_writer, "effects", item_names)
    plot_pollution(pollution_df, summary_writer, "pollution")
    summary_writer.close()

    


def created_tensorboard_layout(df,item_names):
    df = df.groupby('episode')
    layout = {}
    graphs = ["crafted", "inventory", "effects"]


    layout_names = {}
    for name in graphs:
        graph_layout_names = [name + "/" + item for item in item_names]
        layout_names[name] = ["Multiline", graph_layout_names]

    for episode, data in df:
        layout["episode " + str(episode)] = layout_names
    
    return layout

#TENSORBOARD REGEX
#(paperclip_factory|steel|wood|miner|chopper|axe|pickaxe|pollution|paperclip)

#python -m tensorboard.main --logdir=data/simulation_3/plot_data
#python -m tensorboard.main --logdir=data/simulation_3/PPO_1