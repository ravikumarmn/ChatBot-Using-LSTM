import os

import plotly.graph_objects as go

import config
import wandb


def plot(x_axis,y_axis,name = "loss"):
    titles = f"Traning and Validation {name.upper()}"
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=y_axis[0],
        name=F"Training_{name}" 
    ))


    fig.add_trace(go.Scatter(
        x=x_axis,
        y=y_axis[1],
        name=f"Validation_{name}"
    ))

    fig.update_layout(
        title=titles,
        xaxis_title="EPOCHS",
        yaxis_title=f"{name.upper()}",
        legend_title="",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="RebeccaPurple"
        ),
        plot_bgcolor = "white"
    )
    folder_name = f"results/{config.EXPERIMENT_NAME}"
    if not os.path.exists(folder_name):
        print(f"Creating directory {folder_name}")
        os.mkdir(folder_name)
        save_file_path =folder_name+f"/{titles.title()}.png"
        fig.write_image(save_file_path)
        fig.show()

    else:
        save_file_path =folder_name+f"/{titles.title()}.png"
        fig.write_image(save_file_path)
        fig.show()


def init_wandb(params,arg_params):
    wandb.init(
        config=params,
        project="chatbot",
        entity="ravikumarmn",
        name=f'{params["MODEL_NAME"]}_batch_size_{params["BATCH_SIZE"]}_learning_rate_{params["LEARNING_RATE"]}',
        notes = f"Building chatbot using model {params['MODEL_NAME']}.",
        tags=[params['MODEL_NAME']],
        mode=arg_params.wandb)