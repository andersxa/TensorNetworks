#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

parameters = [
        # {'degree': 3, 'max_degree': 8, 'd': 1, 'rank': 6, 'cpd_rank': 100},
        # {'degree': 3, 'max_degree': 8, 'd': 3, 'rank': 12, 'cpd_rank': 100},
        {'degree': 3, 'max_degree': 8, 'd': 7, 'rank': 24, 'cpd_rank': 100},
        # {'degree': 5, 'max_degree': 10, 'd': 1, 'rank': 12, 'cpd_rank': 100},
        {'degree': 5, 'max_degree': 10, 'd': 3, 'rank': 24, 'cpd_rank': 100},
        # {'degree': 5, 'max_degree': 10, 'd': 7, 'rank': 38, 'cpd_rank': 100},
]
all_dfs = []
for params in parameters:
    df = pd.read_csv(f"results_d{params['d']}_deg{params['degree']}_rank{params['rank']}_cpdrank{params['cpd_rank']}_r2.csv")
    # In `name` column replace with the following mapping:
    name_mapping = {
        'Poly.Reg.': 'Poly Regr',
        'CPD': 'CPD',
        'Growing CPD': 'Grow CPD',
        'TT': '(MPO)²',
        'Growing TT': 'Grow (MPO)²',
    }
    # And also sort them to appear in the order of the mapping above
    df['name'] = df['name'].map(name_mapping)
    df['name'] = pd.Categorical(df['name'], categories=name_mapping.values(), ordered=True)
    df = df.sort_values('name')
    df['Num samples'] = df['N']
    df['Model'] = df['name']
    df['Model Degree'] = df['degree']
    all_dfs.append(df)
for params, df in zip(parameters, all_dfs):
    plt.figure(figsize=(8, 6))
    sns.color_palette("tab10")
    sns.lineplot(
        data=df,
        x="Model Degree",
        y="loss",
        style="Num samples",
        hue="Model",
        markers=True,
        errorbar="se",
        palette='tab10'
    )

    plt.title(f"RMSE for problems with dimension {params['d']} and polynomial degree {params['degree']}")
    plt.ylabel("Validation Loss")
    plt.tight_layout()
    # plt.savefig('AFIG0.png')
    plt.tight_layout(pad=1.0)
    #plt.savefig(f"figs/ablation_d{params['d']}_deg{params['degree']}_rank{params['rank']}_cpdrank{params['cpd_rank']}.pdf", bbox_inches='tight', pad_inches=0)
#%%
import seaborn as sns
import matplotlib.pyplot as plt
for params, df in zip(parameters, all_dfs):
    df_N = df[df['N'] == df['N'].max()]
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df_N,
        x="degree",
        y="time",
        hue="name"
    )
    plt.title(f"Training Time")
    plt.xlabel("Degree")
    plt.ylabel("Time (seconds)")
    #Set y axis to log scale
    plt.tight_layout()
    # plt.savefig('AFIG1.png')
for params, df in zip(parameters, all_dfs):
    df_N = df[df['N'] == df['N'].max()].copy()
    df_N['degree_diff'] = np.abs(df_N['best_degree'] - params['degree'])
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df_N,
        x="name",
        y="degree_diff",
    )
    plt.tight_layout()
    # plt.savefig('AFIG2.png')
print(all_dfs[0]['name'].unique())
