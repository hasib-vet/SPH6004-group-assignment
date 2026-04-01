# %%
#importing relevant packages

import numpy as np
import pandas as pd

# %%
# reading the cohort file from step 1
df_cohort_step2 = pd.read_csv("/Users/abdullahhasib/Documents/application file/NUS/courses/semester 2/SPH6004 Advanced Statistical Learning/group assignment/SPH6004-group-assignment/cohort.csv")

df_cohort_step2.head()

# %%
# identify the value count for the target variable
df_cohort_step2["icu_death_flag"].value_counts()

# %%
#description of the length of stay variable
df_cohort_step2["icu_los_hours"].describe()

# %%
#creating a new column label_Y48 to indicate alive or dead in 48 hours. 

df_cohort_step2["label_Y48"] = (
    (df_cohort_step2["icu_death_flag"] == 0) &
    (df_cohort_step2["icu_los_hours"] <= 48)
).astype(int)

df_cohort_step2["label_Y48"].value_counts()

# %%
#same thing but for 72 hours

df_cohort_step2["label_Y72"] = (
    (df_cohort_step2["icu_death_flag"] == 0) &
    (df_cohort_step2["icu_los_hours"] <= 72)
).astype(int)

df_cohort_step2["label_Y72"].value_counts()

# %%
#exporting the cohort file into working directory for step 3
df_cohort_step2.to_csv("cohort_step_2.csv", index=False)


