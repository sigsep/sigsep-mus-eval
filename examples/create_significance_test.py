import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
from matplotlib import gridspec
import scikit_posthocs as sp
import museval

sns.set()
sns.set_context("notebook")

metrics = ['SDR']
targets = ['vocals', 'drums', 'bass', 'other']

oracles = [
    'IBM1', 'IBM2', 'IRM1', 'IRM2', 'MWF'
]

comparisons = museval.MethodsStore()
comparisons.add_sisec18()
agg_df = comparisons.agg_frames_scores()

# Convert to Pandas Dataframes
agg_df['oracle'] = agg_df.method.isin(oracles)

# Get sorting keys (sorted by median of SDR:vocals)
df_sort_by = agg_df[
    (agg_df.metric == "SDR") &
    (agg_df.target == "vocals")
]

methods_by_sdr = df_sort_by.score.groupby(
    df_sort_by.method
).median().sort_values().index.tolist()


f = plt.figure(figsize=(22, 20))
# resort them by median SDR
# Get sorting keys (sorted by median of SDR:vocals score)
df_voc = agg_df[(agg_df.target == 'vocals') & (agg_df.metric == "SAR")]

targets_by_voc_sdr = df_voc.score.groupby(
    df_voc.method
).median().sort_values().index.tolist()

# prepare the pairwise statistics
pc_voc = sp.posthoc_conover(df_voc, val_col='score', group_col='method', sort=True)
print(pc_voc)

f = plt.figure(figsize=(10, 10))
# Format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
cmap = ['1', '#ff2626',  '#ffffff', '#fcbdbd', '#ff7272']
heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5',
                'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.90, 0.35, 0.04, 0.3]}
sp.sign_plot(pc_voc, **heatmap_args)

f.tight_layout()
f.savefig(
    "pairwise_vocals.pdf",
    bbox_inches='tight',
)
