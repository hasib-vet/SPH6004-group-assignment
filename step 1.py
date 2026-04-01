# %%
# Importing relevant package
import pandas as pd
import numpy as np

# %%
# Bringing the dataset

df_static = pd.read_csv("/Users/abdullahhasib/Library/CloudStorage/OneDrive-SharedLibraries-NationalUniversityofSingapore/Ansel Lim - SPH6004_group/data/MIMIC-IV-static(Group Assignment).csv")
df_static

# %%
# checking ICU types

first_careunit = df_static["first_careunit"].unique()
first_careunit

# %%
### https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isin.html
#dropping non ICU units

EXCLUDED_CAREUNITS = ["Neuro Intermediate", "Neuro Stepdown"]
df_INCLUDED_ICU = df_static[~df_static["first_careunit"].isin(EXCLUDED_CAREUNITS)]
df_INCLUDED_ICU

# %%
#converting intime to datetime format so that we can sort time-based operations on it later on
df_INCLUDED_ICU["intime"].dtype

# %%
#checking the format of the "intime" column 
df_INCLUDED_ICU["intime"] = pd.to_datetime(df_INCLUDED_ICU["intime"])
df_INCLUDED_ICU["intime"].dtype

# %%
# we only intend to consider the first ICU stay of each patient so we sorting the dataframe by "intime" in ascending order to make sure the first arrival is on top
df_intime = df_INCLUDED_ICU.sort_values("intime")

df_intime.head()

# %%
### https://www.youtube.com/watch?v=txMdrV1Ut64
# after that we group the dataframe by "subject_id" and bring them to the first row of each group
df_intime_new = df_intime.groupby("subject_id").first().reset_index()

df_intime_new.head()

# %%
# checking result - the number of subject_id should be the same as the length of the new dataframe as we collapsed them
print(f"Unique patients: {df_intime_new['subject_id'].nunique()}")
print(f"Total rows: {len(df_intime_new)}")

# %%
# checking the value of icu_los_hours 
df_intime_new["icu_los_hours"]

# %%
#include only icu_los_hours which corresponds the value of more than 24 hours
df_cohort = df_intime_new[df_intime_new["icu_los_hours"] > 24]

df_cohort

# %%
### https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
#making sure the minimum value of icu_los_hours is more than 24 hours
df_cohort['icu_los_hours'].min()

# %%
#same like before but we do it now for dead patients
#some patients maybe dead before 24 hours in icu but discharge after hour 24, e.g. die at hour 20 but discharge at hour 30

#checking the format of deathtime column
df_cohort["deathtime"].dtype

# %%
#converting deathtime to datetime format
df_cohort["deathtime"] = pd.to_datetime(df_cohort["deathtime"])

print(df_cohort["deathtime"].dtype)

# %%
#copnverting to hours with 3600 and adding a new column "hours_to_death" to the dataframe
df_cohort["hours_to_death"] = (
    df_cohort["deathtime"] - df_cohort["intime"]
).dt.total_seconds() / 3600

# %%
df_cohort["hours_to_death"]

# %%
#inclding only patietns who died after 24 hours in ICU
died_before_24 = df_cohort["hours_to_death"] < 24

died_before_24.unique()

# %%
#make it back into the dataframe
df_cohort = df_cohort[~died_before_24]
df_cohort

# %%
### https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
# exporting to working directory

df_cohort.to_csv("cohort.csv", index=False)


