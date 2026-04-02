# %%
#import relevant pavckhages
import pandas as pd
import numpy as np

# %%
### https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html #for usecols to load cohort

df_cohort_step_4 = pd.read_csv("cohort_step_2.csv", usecols=["stay_id", "intime"])

df_cohort_step_4

# %%
#checking after converting intime
df_cohort_step_4["intime"] = pd.to_datetime(df_cohort_step_4["intime"])

df_cohort_step_4["intime"].dtype

# %%
#defining stay_id set for filtering
cohort_stay_ids = set(df_cohort_step_4["stay_id"])
chunks = []
cohort_stay_ids

# %%
### https://gist.github.com/cengizhancaliskan/e2aff781378793454b7615cb02745fc7

#reading lare file in chunks and filtering by stay_id

filename = "/Users/abdullahhasib/Library/CloudStorage/OneDrive-SharedLibraries-NationalUniversityofSingapore/Ansel Lim - SPH6004_group/data/MIMIC-IV-time_series(Group Assignment).csv"
chunksize = 10 ** 6
chunk_list = []

for chunk in pd.read_csv(filename, chunksize=chunksize, na_values="NULL"):
    chunk = chunk[chunk["stay_id"].isin(cohort_stay_ids)]
    chunk_list.append(chunk)

df_ts = pd.concat(chunk_list, ignore_index=True)
del chunk_list

print(f"Total rows: {len(df_ts)}")

# %%



