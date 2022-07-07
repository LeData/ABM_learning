import matplotlib.pyplot as plt
import pandas as pd
from mesa.batchrunner import batch_run
from model import MatingModel
import seaborn as sns
import numpy as np

density = 0.3
height = 20
width = 10

model_params = {
    "height": height,
    "width": width,
    "it_M": 0.3,
    "it_F": 0.5,
    "num_agents": int(height*width*density),
    "balance": 0.5
}

model_params_extended = {
    "height": 20,
    "width": 10,
    "it_M": np.linspace(0.2, 0.8, 4),
    "it_F": np.linspace(0.5, 1.3, 4),
    "num_agents": int(height*width*density),
    "balance": np.linspace(.2, .8, 6),
}

simple = False

if __name__ == "__main__":
    results_s = batch_run(
        MatingModel,
        parameters=model_params if simple else model_params_extended,
        iterations=60,
        max_steps=1000,
        number_processes=None,
        data_collection_period=1,
        display_progress=True,
    )

    df = pd.DataFrame(results_s)
    df.to_csv(f"outputs/single_run_d03_i60{'' if simple else '_extended'}.csv")

    titles = {
        "Incel": f"Proportion of incels within singles,",
        "Single": f"Proportion of singles,",
        "Avg_rej": f"Average number of rejections,"
    }

    if simple:

        for var in ["Single", "Incel", "Avg_rej"]:
            plot_df = (
                df
                .groupby([*model_params.keys(), "iteration"])
                .agg({f"{var}_{sex}": min for sex in MatingModel.sexes})
                .reset_index()
                .drop("iteration", axis=1)
            )
            title = titles[var] + f"\n balance = {model_params['balance']}, area = {height*width}, density = {density}"
            fig, ax = plt.subplots(figsize=(20, 10))
            ax = sns.scatterplot(data=plot_df, x=f"{var}_M", y=f"{var}_F", hue="num_agents")
            ax.set_title(title)
            fig.savefig(f"outputs/{var}.png")
            plt.close(fig)