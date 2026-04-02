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

cohort_stay_ids

# %%
### https://gist.github.com/cengizhancaliskan/e2aff781378793454b7615cb02745fc7
#reading in chunks and filtering by stay_id

filename = "/Users/abdullahhasib/Library/CloudStorage/OneDrive-SharedLibraries-NationalUniversityofSingapore/Ansel Lim - SPH6004_group/data/MIMIC-IV-time_series(Group Assignment).csv"
chunksize = 10 ** 6
chunk_list = []

for chunk in pd.read_csv(filename, chunksize=chunksize, na_values="NULL"):
    chunk = chunk[chunk["stay_id"].isin(cohort_stay_ids)]
    chunk_list.append(chunk)

df_ts = pd.concat(chunk_list, ignore_index=True)
del chunk_list

df_ts

# %%



# %%
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html

# changing hour_ts to datetime for calculation
df_ts["hour_ts"] = pd.to_datetime(df_ts["hour_ts"])

df_ts["hour_ts"].dtype

# %%
# merging df_ts with df_cohort_step_4 to get intime for each stay_id
df_ts = df_ts.merge(df_cohort_step_4[["stay_id", "intime"]], on="stay_id", how="left")

df_ts.head()

# %%
#converting hour_ts and intime by 3600 to get relative hour
df_ts["relative_hour"] = (
    df_ts["hour_ts"] - df_ts["intime"]
).dt.total_seconds() / 3600

df_ts.head()

# %%
#dropping hour_ts and intime
df_ts = df_ts.drop(columns=["hour_ts", "intime"])

df_ts["relative_hour"].describe()

# %%
# keeping only data within observation window [-3h to +24h]

df_ts_window = df_ts[(df_ts["relative_hour"] >= -3) &(df_ts["relative_hour"] <= 24)].copy()

#checking before and after filtering by observation window
print(f"Before filter : {len(df_ts):,} rows")
print(f"After filter  : {len(df_ts_window):,} rows")
print(f"Unique stays  : {df_ts_window['stay_id'].nunique():,}")

# %% [markdown]
# # calculating measurement coverage for each variable
# # to determine which variables are dense vs sparse
# # Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
# 
# all_vars = [col for col in df_ts_window.columns
#             if col not in ["stay_id", "relative_hour"]]
# 
# coverage = {}
# for var in all_vars:
#     coverage[var] = df_ts_window[var].notna().mean() * 100
# 
# df_coverage = pd.DataFrame.from_dict(
#     coverage, orient="index", columns=["coverage_%"]
# ).sort_values("coverage_%", ascending=False)
# 
# print(df_coverage)

# %%
### https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html
#Determining which variables are dense vs sparse based on coverage
#checking how often each variable is measured

coverage = df_ts_window.notna().mean() * 100

coverage

# %%
# sorting the variables based on frequency 

coverage.sort_values(ascending=False)

# %%
# classifying variables based on coverage
# dense  = coverage > 50%
# sparse = coverage <= 50%

DENSE_VARS  = ["ventilation_flag", "hr", "rr", "sofa_cardio", "urine_output"]


SPARSE_VARS = ["gcs", "sofa_cns", "temp", "sofa_renal", "sodium", "vasopressor_dose", "map",  
               "creatinine", "platelets", "bun", "sofa_coag", "lactate", "sofa_resp", "bilirubin",
               "sofa_liver", "fluid_input", "wbc"]

#dropping spo2 and intr as they are 0 
df_ts_window = df_ts_window.drop(columns=["spo2", "inr"], errors="ignore")

df_ts_window.columns

# %% [markdown]
# # classifying variables based on coverage
# # dense  = coverage > 50% (measured continuously)
# # sparse = coverage <= 50% (measured intermittently)
# # drop   = coverage = 0% (completely missing, e.g. spo2, inr)
# # Reference: eda_dense_sparse.py from group pipeline
# 
# DENSE_VARS = df_coverage[df_coverage["coverage_%"] > 50].index.tolist()
# SPARSE_VARS = df_coverage[
#     (df_coverage["coverage_%"] > 0) &
#     (df_coverage["coverage_%"] <= 50)
# ].index.tolist()
# DROP_VARS = df_coverage[df_coverage["coverage_%"] == 0].index.tolist()
# 
# print(f"Dense  ({len(DENSE_VARS)})  : {DENSE_VARS}")
# print(f"Sparse ({len(SPARSE_VARS)}) : {SPARSE_VARS}")
# print(f"Drop   ({len(DROP_VARS)})   : {DROP_VARS}")

# %% [markdown]
# # aggregating dense variables using 7 statistics
# # dense variables are measured frequently enough to estimate trend (slope)
# # Reference: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.GroupBy.agg.html
# # Reference: https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
# 
# def slope(series):
#     valid = series.dropna()
#     if len(valid) < 2:
#         return np.nan
#     x = np.arange(len(valid))
#     return np.polyfit(x, valid.values, 1)[0]
# 
# dense_frames = []
# for var in DENSE_VARS:
#     if var not in df_ts_window.columns:
#         continue
#     grp = df_ts_window.groupby("stay_id")[var]
#     agg = grp.agg(mean="mean", min="min", max="max",
#                   std="std", last="last", count="count")
#     agg["slope"] = grp.apply(slope)
#     agg.columns = [f"{var}_{stat}" for stat in agg.columns]
#     dense_frames.append(agg)
#     print(f"Done: {var}")
# 
# df_dense_agg = pd.concat(dense_frames, axis=1).reset_index()
# print(f"\nShape: {df_dense_agg.shape}")

# %%
# defining slope function to measure variable trend as clinically important
# slope > 0 = increasing, slope < 0 = decreasing
# Reference: https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html

def slope(series):
    valid = series.dropna()
    if len(valid) < 2:
        return np.nan
    x = np.arange(len(valid))
    return np.polyfit(x, valid.values, 1)[0]

# %%
#dense variable aggregation with 7 statistical features (mean, min, max, std, last value, count, slope)
#aggregating heart rate
hr_agg = df_ts_window.groupby("stay_id")["hr"].agg(
    hr_mean="mean",
    hr_min="min",
    hr_max="max",
    hr_std="std",
    hr_last="last",
    hr_count="count"
).reset_index()

hr_agg["hr_slope"] = df_ts_window.groupby("stay_id")["hr"].apply(slope).values

hr_agg.head()

# %%
# adding binary column for whether heart rate was once measured during the observation window 

hr_agg["hr_measured"] = (hr_agg["hr_count"] > 0).astype(int)

hr_agg.head()

# %%
# aggregating respiratory rate

rr_agg = df_ts_window.groupby("stay_id")["rr"].agg(
    rr_mean="mean",
    rr_min="min",
    rr_max="max",
    rr_std="std",
    rr_last="last",
    rr_count="count"
).reset_index()

rr_agg["rr_slope"] = df_ts_window.groupby("stay_id")["rr"].apply(slope).values

rr_agg.head()

# %%
# adding binary column for whether heart rate was once measured during the observation window 

rr_agg["rr_measured"] = (rr_agg["rr_count"] > 0).astype(int)

rr_agg.head()

# %%
#merging them based on stay_id
df_dense_agg = hr_agg.merge(rr_agg, on="stay_id", how="left")

df_dense_agg.head()

# %%
# aggregating ventilation flag

ventilation_agg = df_ts_window.groupby("stay_id")["ventilation_flag"].agg(
    ventilation_flag_mean="mean",
    ventilation_flag_min="min",
    ventilation_flag_max="max",
    ventilation_flag_std="std",
    ventilation_flag_last="last",
    ventilation_flag_count="count"
).reset_index()

ventilation_agg["ventilation_flag_slope"] = df_ts_window.groupby("stay_id")["ventilation_flag"].apply(slope).values

ventilation_agg.head()

# %%
ventilation_agg["ventilation_flag_measured"] = (ventilation_agg["ventilation_flag_count"] > 0).astype(int)

ventilation_agg.head()

# %%
df_dense_agg = df_dense_agg.merge(ventilation_agg, on="stay_id", how="left")

df_dense_agg.head()

# %%
# aggregating urine output

urine_agg = df_ts_window.groupby("stay_id")["urine_output"].agg(
    urine_output_mean="mean",
    urine_output_min="min",
    urine_output_max="max",
    urine_output_std="std",
    urine_output_last="last",
    urine_output_count="count"
).reset_index()

urine_agg["urine_output_slope"] = df_ts_window.groupby("stay_id")["urine_output"].apply(slope).values

urine_agg.head()

# %%
urine_agg["urine_output_measured"] = (urine_agg["urine_output_count"] > 0).astype(int)

urine_agg.head()

# %%
df_dense_agg = df_dense_agg.merge(urine_agg, on="stay_id", how="left")

df_dense_agg.head()

# %%
# aggregating sofa cardiovascular score

sofa_cardio_agg = df_ts_window.groupby("stay_id")["sofa_cardio"].agg(
    sofa_cardio_mean="mean",
    sofa_cardio_min="min",
    sofa_cardio_max="max",
    sofa_cardio_std="std",
    sofa_cardio_last="last",
    sofa_cardio_count="count"
).reset_index()

sofa_cardio_agg["sofa_cardio_slope"] = df_ts_window.groupby("stay_id")["sofa_cardio"].apply(slope).values

sofa_cardio_agg.head()

# %%
sofa_cardio_agg["sofa_cardio_measured"] = (sofa_cardio_agg["sofa_cardio_count"] > 0).astype(int)

sofa_cardio_agg.head()

# %%
#all together in one final aggregated dataframe
df_dense_agg = df_dense_agg.merge(sofa_cardio_agg, on="stay_id", how="left")

df_dense_agg.head()

# %%
#checking final shape
df_dense_agg.shape

# %%


# %%
#aggregating sparse variables with 5 statistical features
#aggregating glasgow coma scale (gcs)
gcs_agg = df_ts_window.groupby("stay_id")["gcs"].agg(
    gcs_mean="mean",
    gcs_min="min",
    gcs_max="max",
    gcs_last="last",
    gcs_count="count"
).reset_index()

gcs_agg.head()

# %%
gcs_agg["gcs_measured"] = (gcs_agg["gcs_count"] > 0).astype(int)

gcs_agg.head()

# %%
# aggregating temperature (temp)

temp_agg = df_ts_window.groupby("stay_id")["temp"].agg(
    temp_mean="mean",
    temp_min="min",
    temp_max="max",
    temp_last="last",
    temp_count="count"
).reset_index()

temp_agg.head()

# %%
temp_agg["temp_measured"] = (temp_agg["temp_count"] > 0).astype(int)

temp_agg.head()

# %%
df_sparse_agg = gcs_agg.merge(temp_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating mean arterial pressure (map)

map_agg = df_ts_window.groupby("stay_id")["map"].agg(
    map_mean="mean",
    map_min="min",
    map_max="max",
    map_last="last",
    map_count="count"
).reset_index()

map_agg.head()

# %%
map_agg["map_measured"] = (map_agg["map_count"] > 0).astype(int)

map_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(map_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating lactate

lactate_agg = df_ts_window.groupby("stay_id")["lactate"].agg(
    lactate_mean="mean",
    lactate_min="min",
    lactate_max="max",
    lactate_last="last",
    lactate_count="count"
).reset_index()

lactate_agg.head()

# %%
lactate_agg["lactate_measured"] = (lactate_agg["lactate_count"] > 0).astype(int)

lactate_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(lactate_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating creatinine

creatinine_agg = df_ts_window.groupby("stay_id")["creatinine"].agg(
    creatinine_mean="mean",
    creatinine_min="min",
    creatinine_max="max",
    creatinine_last="last",
    creatinine_count="count"
).reset_index()

creatinine_agg.head()

# %%
creatinine_agg["creatinine_measured"] = (creatinine_agg["creatinine_count"] > 0).astype(int)

creatinine_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(creatinine_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating bilirubin

bilirubin_agg = df_ts_window.groupby("stay_id")["bilirubin"].agg(
    bilirubin_mean="mean",
    bilirubin_min="min",
    bilirubin_max="max",
    bilirubin_last="last",
    bilirubin_count="count"
).reset_index()

bilirubin_agg.head()

# %%
bilirubin_agg["bilirubin_measured"] = (bilirubin_agg["bilirubin_count"] > 0).astype(int)

bilirubin_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(bilirubin_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating platelets

platelets_agg = df_ts_window.groupby("stay_id")["platelets"].agg(
    platelets_mean="mean",
    platelets_min="min",
    platelets_max="max",
    platelets_last="last",
    platelets_count="count"
).reset_index()

platelets_agg.head()

# %%
platelets_agg["platelets_measured"] = (platelets_agg["platelets_count"] > 0).astype(int)

platelets_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(platelets_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating white blood cells (wbc)

wbc_agg = df_ts_window.groupby("stay_id")["wbc"].agg(
    wbc_mean="mean",
    wbc_min="min",
    wbc_max="max",
    wbc_last="last",
    wbc_count="count"
).reset_index()

wbc_agg

# %%
wbc_agg["wbc_measured"] = (wbc_agg["wbc_count"] > 0).astype(int)

wbc_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(wbc_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating sodium

sodium_agg = df_ts_window.groupby("stay_id")["sodium"].agg(
    sodium_mean="mean",
    sodium_min="min",
    sodium_max="max",
    sodium_last="last",
    sodium_count="count"
).reset_index()

sodium_agg.head()

# %%
sodium_agg["sodium_measured"] = (sodium_agg["sodium_count"] > 0).astype(int)

sodium_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(sodium_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating blood urea nitrogen (bun)

bun_agg = df_ts_window.groupby("stay_id")["bun"].agg(
    bun_mean="mean",
    bun_min="min",
    bun_max="max",
    bun_last="last",
    bun_count="count"
).reset_index()

bun_agg.head()

# %%
bun_agg["bun_measured"] = (bun_agg["bun_count"] > 0).astype(int)

bun_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(bun_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating vasopressor dose

vasopressor_agg = df_ts_window.groupby("stay_id")["vasopressor_dose"].agg(
    vasopressor_dose_mean="mean",
    vasopressor_dose_min="min",
    vasopressor_dose_max="max",
    vasopressor_dose_last="last",
    vasopressor_dose_count="count"
).reset_index()

vasopressor_agg.head()

# %%
vasopressor_agg["vasopressor_dose_measured"] = (vasopressor_agg["vasopressor_dose_count"] > 0).astype(int)

vasopressor_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(vasopressor_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating fluid input

fluid_agg = df_ts_window.groupby("stay_id")["fluid_input"].agg(
    fluid_input_mean="mean",
    fluid_input_min="min",
    fluid_input_max="max",
    fluid_input_last="last",
    fluid_input_count="count"
).reset_index()

fluid_agg

# %%
fluid_agg["fluid_input_measured"] = (fluid_agg["fluid_input_count"] > 0).astype(int)

fluid_agg

# %%
df_sparse_agg = df_sparse_agg.merge(fluid_agg, on="stay_id", how="left")

df_sparse_agg

# %%
# aggregating sofa respiratory score (sofa_resp)

sofa_resp_agg = df_ts_window.groupby("stay_id")["sofa_resp"].agg(
    sofa_resp_mean="mean", sofa_resp_min="min", sofa_resp_max="max",
    sofa_resp_last="last", sofa_resp_count="count"
).reset_index()

sofa_resp_agg.head()

# %%
sofa_resp_agg["sofa_resp_measured"] = (sofa_resp_agg["sofa_resp_count"] > 0).astype(int)

sofa_resp_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(sofa_resp_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating sofa renal score (sofa_renal)

sofa_renal_agg = df_ts_window.groupby("stay_id")["sofa_renal"].agg(
    sofa_renal_mean="mean", sofa_renal_min="min", sofa_renal_max="max",
    sofa_renal_last="last", sofa_renal_count="count"
).reset_index()

sofa_renal_agg.head()

# %%
sofa_renal_agg["sofa_renal_measured"] = (sofa_renal_agg["sofa_renal_count"] > 0).astype(int)

sofa_renal_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(sofa_renal_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating sofa liver score (sofa_liver)

sofa_liver_agg = df_ts_window.groupby("stay_id")["sofa_liver"].agg(
    sofa_liver_mean="mean", sofa_liver_min="min", sofa_liver_max="max",
    sofa_liver_last="last", sofa_liver_count="count"
).reset_index()

sofa_liver_agg.head()

# %%
sofa_liver_agg["sofa_liver_measured"]       = (sofa_liver_agg["sofa_liver_count"] > 0).astype(int)

sofa_liver_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(sofa_liver_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating sofa coagulation score (sofa_coag)

sofa_coag_agg = df_ts_window.groupby("stay_id")["sofa_coag"].agg(
    sofa_coag_mean="mean", sofa_coag_min="min", sofa_coag_max="max",
    sofa_coag_last="last", sofa_coag_count="count"
).reset_index()

sofa_coag_agg.head()

# %%
sofa_coag_agg["sofa_coag_measured"] = (sofa_coag_agg["sofa_coag_count"] > 0).astype(int)

sofa_coag_agg.head()

# %%
df_sparse_agg = df_sparse_agg.merge(sofa_coag_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
# aggregating sofa central nervous system score (sofa_cns)

sofa_cns_agg = df_ts_window.groupby("stay_id")["sofa_cns"].agg(
    sofa_cns_mean="mean", sofa_cns_min="min", sofa_cns_max="max",
    sofa_cns_last="last", sofa_cns_count="count"
).reset_index()

sofa_cns_agg.head()

# %%
sofa_cns_agg["sofa_cns_measured"] = (sofa_cns_agg["sofa_cns_count"] > 0).astype(int)

sofa_cns_agg.head()

# %%
#all together in for sparse variables

df_sparse_agg = df_sparse_agg.merge(sofa_cns_agg, on="stay_id", how="left")

df_sparse_agg.head()

# %%
#checking final shape of sparse aggregated dataframe

df_sparse_agg.shape

# %% [markdown]
# # aggregating sparse variables using 5 statistics only
# # std and slope excluded because too few measurements to be reliable
# # Reference: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.GroupBy.agg.html
# 
# sparse_frames = []
# for var in SPARSE_VARS:
#     if var not in df_ts_window.columns:
#         continue
#     grp = df_ts_window.groupby("stay_id")[var]
#     agg = grp.agg(mean="mean", min="min", max="max",
#                   last="last", count="count")
#     agg.columns = [f"{var}_{stat}" for stat in agg.columns]
#     sparse_frames.append(agg)
#     print(f"Done: {var}")
# 
# df_sparse_agg = pd.concat(sparse_frames, axis=1).reset_index()
# print(f"\nShape: {df_sparse_agg.shape}")

# %% [markdown]
# # merging dense and sparse aggregations
# # adding missingness flags: 1 if variable was ever measured, 0 if not
# # Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
# 
# df_ts_agg = df_dense_agg.merge(df_sparse_agg, on="stay_id", how="left")
# 
# for var in DENSE_VARS + SPARSE_VARS:
#     count_col = f"{var}_count"
#     if count_col in df_ts_agg.columns:
#         df_ts_agg[f"{var}_measured"] = (df_ts_agg[count_col] > 0).astype(int)
# 
# print(f"Final shape: {df_ts_agg.shape}")
# print(f"Expected from pipeline log: 40,457 rows x 142 columns")
# df_ts_agg.head()

# %%
# merging dense and sparse aggregations 

df_ts_agg = df_dense_agg.merge(df_sparse_agg, on="stay_id", how="left")

df_ts_agg

# %% [markdown]
# # adding missingness flag for each variable
# # 1 = variable was measured at least once during observation window
# # 0 = variable was never measured
# # Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
# 
# all_vars = DENSE_VARS + SPARSE_VARS
# 
# for var in all_vars:
#     count_col = f"{var}_count"
#     if count_col in df_ts_agg.columns:
#         df_ts_agg[f"{var}_measured"] = (df_ts_agg[count_col] > 0).astype(int)

# %% [markdown]
# # adding missingness flags for each variable
# # 1 = measured at least once, 0 = never measured
# # Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
# 
# 
# df_ts_agg["rr_measured"]               = (df_ts_agg["rr_count"] > 0).astype(int)
# df_ts_agg["ventilation_flag_measured"] = (df_ts_agg["ventilation_flag_count"] > 0).astype(int)
# df_ts_agg["urine_output_measured"]     = (df_ts_agg["urine_output_count"] > 0).astype(int)
# df_ts_agg["sofa_cardio_measured"]      = (df_ts_agg["sofa_cardio_count"] > 0).astype(int)
# df_ts_agg["gcs_measured"]              = (df_ts_agg["gcs_count"] > 0).astype(int)
# df_ts_agg["temp_measured"]             = (df_ts_agg["temp_count"] > 0).astype(int)
# df_ts_agg["map_measured"]              = (df_ts_agg["map_count"] > 0).astype(int)
# df_ts_agg["lactate_measured"]          = (df_ts_agg["lactate_count"] > 0).astype(int)
# df_ts_agg["creatinine_measured"]       = (df_ts_agg["creatinine_count"] > 0).astype(int)
# df_ts_agg["bilirubin_measured"]        = (df_ts_agg["bilirubin_count"] > 0).astype(int)
# df_ts_agg["platelets_measured"]        = (df_ts_agg["platelets_count"] > 0).astype(int)
# df_ts_agg["wbc_measured"]              = (df_ts_agg["wbc_count"] > 0).astype(int)
# df_ts_agg["sodium_measured"]           = (df_ts_agg["sodium_count"] > 0).astype(int)
# df_ts_agg["bun_measured"]              = (df_ts_agg["bun_count"] > 0).astype(int)
# df_ts_agg["vasopressor_dose_measured"] = (df_ts_agg["vasopressor_dose_count"] > 0).astype(int)
# df_ts_agg["fluid_input_measured"]      = (df_ts_agg["fluid_input_count"] > 0).astype(int)
# df_ts_agg["sofa_resp_measured"]        = (df_ts_agg["sofa_resp_count"] > 0).astype(int)
# df_ts_agg["sofa_renal_measured"]       = (df_ts_agg["sofa_renal_count"] > 0).astype(int)
# df_ts_agg["sofa_liver_measured"]       = (df_ts_agg["sofa_liver_count"] > 0).astype(int)
# df_ts_agg["sofa_coag_measured"]        = (df_ts_agg["sofa_coag_count"] > 0).astype(int)
# df_ts_agg["sofa_cns_measured"]         = (df_ts_agg["sofa_cns_count"] > 0).astype(int)
# 
# print(f"Final shape: {df_ts_agg.shape}")

# %%
# checking for duplicate columns
print([col for col in df_ts_agg.columns if df_ts_agg.columns.tolist().count(col) > 1])

# checking all columns
print(df_ts_agg.shape)
print(df_ts_agg.columns.tolist())

# checking sofa_liver coverage
print(df_ts_window["sofa_liver"].notna().mean() * 100)

# %%
# exporting time series aggregated features
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html

df_ts_agg.to_csv("timeseries_agg.csv", index=False)

print(f"Saved: {df_ts_agg.shape[0]:,} rows x {df_ts_agg.shape[1]} columns")


