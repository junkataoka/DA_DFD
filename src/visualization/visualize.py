#%%
import matplotlib.pyplot as plt
import os
import numpy as np
# %%
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pandas as pd

def visualize(tar_y, tar_output_trained, tar_output_pretrained, log, i):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        dummy = np.arange(start=1, stop=len(tar_y)+1)
        ax.bar(height=tar_y, x=dummy, label="Target recipe profile", alpha=0.5)
        ax.plot(dummy, tar_output_trained, "r-*", label="Domain-adapted")
        ax.plot(dummy, tar_output_pretrained, "g-o", label="Source-only") 
        ax.set_ylabel(r"Temperature ($^\circ C$)", fontsize=12)
        ax.set_xlabel("Heating zones", fontsize=12)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig(f"./models/{log}/result_fig_sample{i+1}.png", dpi=300)

        mse_trained = (np.square(tar_y - tar_output_trained)).mean()
        mse_pretrained = (np.square(tar_y - tar_output_pretrained)).mean()
        abs_error_trained = (abs(tar_y - tar_output_trained)).mean()
        abs_error_pretrained = (abs(tar_y - tar_output_pretrained)).mean()
        out = pd.DataFrame({
                "MSE_pretrained": [mse_pretrained],
                "MSE_trained": [mse_trained],
                "abs_error_pretrained": [abs_error_pretrained],
                "abs_error_trained": [abs_error_trained],
                        })
        out.to_csv(f"./models/{log}/result_sample{i+1}.csv")


