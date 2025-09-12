#%%
from glob import glob
import pandas as pd
import numpy as np

# 1) Load all result files
paths = glob("./results/*.csv")

df_list = []
for p in paths:
    df = pd.read_csv(p)
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

# 2) Ensure numeric types for key columns
for col in ["val_r2", "val_accuracy"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 3) Determine task type and create the metric column
# Classification if val_accuracy is not NaN. Otherwise regression.
df["is_classification"] = df["val_accuracy"].notna()
df["metric"] = np.where(df["is_classification"], df["val_accuracy"], df["val_r2"])

# 4) Aggregate over seeds
# Group by stable identifiers. Adjust this list if you need more fields kept distinct.
group_cols = [c for c in ["dataset", "N", "r", "lin_dim", "model_type"] if c in df.columns]

agg = (
    df.groupby(group_cols, dropna=False)
      .agg(
          n_seeds=("seed", "nunique"),
          is_classification=("is_classification", "any"),
          metric_mean=("metric", "mean"),
          metric_sem=("metric", "sem"),
          num_params=("num_params", "first"),
      )
      .reset_index()
      .sort_values(group_cols)
)
#%%
# 5) Pivot so that each dataset gets its own metric column
pivoted = agg.pivot(
    index=[c for c in group_cols if c != "dataset"],
    columns="dataset",
    values="metric_mean"
).reset_index()

# If you also want SEM columns for each dataset:
pivoted_sem = agg.pivot(
    index=[c for c in group_cols if c != "dataset"],
    columns="dataset",
    values="metric_sem"
).reset_index()

# Add num_params back to the pivoted dataframe by merging with agg data
# Since num_params should be the same for all datasets with the same configuration,
# we can get it from the first occurrence for each group
# We need to be careful with NaN values in lin_dim - handle them separately
index_cols = [c for c in group_cols if c != "dataset"]
num_params_mapping = agg.groupby(index_cols, dropna=False)["num_params"].first().reset_index()
pivoted = pivoted.merge(num_params_mapping, on=index_cols, how="left")

# Sort by model type
model_order = [
    "tt", "tt_type1", "tt_lin", "tt_lin_type1",
    "tt_cumsum", "tt_cumsum_type1", "cpd", "cpd_type1"
]

# Make model_type categorical with this order
pivoted["model_type"] = pd.Categorical(
    pivoted["model_type"],
    categories=model_order,
    ordered=True
)

# Now sort hierarchically by model_type, N, r, lin_dim
pivoted = pivoted.sort_values(
    by=["model_type", "N", "r", "lin_dim"],
    ascending=[True, True, True, True]
).reset_index(drop=True)

# TODO: Keep num_params in the print.
pd.set_option('display.max_rows', None)  # Show all rows
print("=== Regression Results ===")
print(pivoted.to_string(index=False))

#%%
# Split into regression and classification for better readability
# Then sort by metric_mean descending and print all rows
regression_df = agg[~agg["is_classification"]].drop(columns=["is_classification"]).sort_values("metric_mean", ascending=False)
classification_df = agg[agg["is_classification"]].drop(columns=["is_classification"]).sort_values("metric_mean", ascending=False)
pd.set_option('display.max_rows', None)  # Show all rows
print("=== Regression Results ===")
print(regression_df.to_string(index=False))
print("\n=== Classification Results ===")
print(classification_df.to_string(index=False))
# %%
