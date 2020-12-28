import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import museval

comparisons = museval.MethodStore()
comparisons.add_sisec18()
agg_df = comparisons.agg_frames_scores()

sns.set()
sns.set_context("notebook")

metrics = ['SDR']
selected_targets = ['vocals', 'drums', 'bass', 'other']
oracles = [
    'IBM1', 'IBM2', 'IRM1', 'IRM2', 'MWF', 'IMSK'
]

# Convert to Pandas Dataframes
agg_df['oracle'] = agg_df.method.isin(oracles)
agg_df = agg_df[agg_df.target.isin(selected_targets)].dropna()

# Get sorting keys (sorted by median of SDR:vocals)
df_sort_by = agg_df[
    (agg_df.metric == "SDR") &
    (agg_df.target == "vocals")
]

methods_by_sdr = df_sort_by.score.groupby(
    df_sort_by.method
).median().sort_values().index.tolist()

# df = df[df.target == "vocals"]
g = sns.FacetGrid(
    agg_df,
    row="target",
    col="metric",
    row_order=selected_targets,
    col_order=metrics,
    size=4,
    sharex=False,
    aspect=3
)
g = (g.map(
    sns.boxplot,
    "score",
    "method",
    "oracle",
    orient='h',
    order=methods_by_sdr[::-1],
    hue_order=[True, False],
    showfliers=False,
    notch=True
))

g.fig.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0.1)
g.fig.savefig(
    "boxplot.pdf",
    bbox_inches='tight',
)
