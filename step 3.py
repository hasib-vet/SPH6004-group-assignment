# %%
import pandas as pd

# %%
df_cohort_step3 = pd.read_csv("cohort_step_2.csv")

df_cohort_step3.head()

# %%
# checking all available columns

df_cohort_step3.columns.tolist()

# %%
Identifier = ['subject_id',
 'hadm_id',
 'stay_id',
 'last_careunit',
 'icu_los_hours',
 'outtime',
 'los',
 'deathtime',
 'icu_death_flag',
 'hospital_expire_flag',
 'hours_to_death']

Demographics = ['insurance',
 'language',
 'race',
 'marital_status',
 'age',
 'gender',
 'first_careunit']

label = ['label_Y48', 'label_Y72']

static_features = ['sofa2_respiration_24h_max',
 'sofa2_cardiovascular_24h_max',
 'sofa2_coagulation_24h_max',
 'sofa2_liver_24h_max',
 'sofa2_renal_24h_max',
 'sofa2_cns_24h_max',
 'heart_rate_min',
 'heart_rate_max',
 'heart_rate_mean',
 'sbp_min',
 'sbp_max',
 'sbp_mean',
 'dbp_min',
 'dbp_max',
 'dbp_mean',
 'mbp_min',
 'mbp_max',
 'mbp_mean',
 'resp_rate_min',
 'resp_rate_max',
 'resp_rate_mean',
 'temperature_min',
 'temperature_max',
 'temperature_mean',
 'spo2_min',
 'spo2_max',
 'spo2_mean',
 'glucose_min',
 'glucose_max',
 'glucose_mean',
 'gcs_min',
 'gcs_motor',
 'gcs_verbal',
 'gcs_eyes',
 'gcs_unable',
 'hematocrit_min',
 'hematocrit_max',
 'hemoglobin_min',
 'hemoglobin_max',
 'platelets_min',
 'platelets_max',
 'wbc_min',
 'wbc_max',
 'albumin_min',
 'albumin_max',
 'globulin_min',
 'globulin_max',
 'total_protein_min',
 'total_protein_max',
 'aniongap_min',
 'aniongap_max',
 'bicarbonate_min',
 'bicarbonate_max',
 'bun_min',
 'bun_max',
 'calcium_min',
 'calcium_max',
 'chloride_min',
 'chloride_max',
 'creatinine_min',
 'creatinine_max',
 'sodium_min',
 'sodium_max',
 'potassium_min',
 'potassium_max',
 'abs_basophils_min',
 'abs_basophils_max',
 'abs_eosinophils_min',
 'abs_eosinophils_max',
 'abs_lymphocytes_min',
 'abs_lymphocytes_max',
 'abs_monocytes_min',
 'abs_monocytes_max',
 'abs_neutrophils_min',
 'abs_neutrophils_max',
 'atyps_min',
 'atyps_max',
 'bands_min',
 'bands_max',
 'imm_granulocytes_min',
 'imm_granulocytes_max',
 'metas_min',
 'metas_max',
 'nrbc_min',
 'nrbc_max',
 'd_dimer_min',
 'd_dimer_max',
 'fibrinogen_min',
 'fibrinogen_max',
 'thrombin_min',
 'thrombin_max',
 'inr_min',
 'inr_max',
 'pt_min',
 'pt_max',
 'ptt_min',
 'ptt_max',
 'alt_min',
 'alt_max',
 'alp_min',
 'alp_max',
 'ast_min',
 'ast_max',
 'amylase_min',
 'amylase_max',
 'bilirubin_total_min',
 'bilirubin_total_max',
 'bilirubin_direct_min',
 'bilirubin_direct_max',
 'bilirubin_indirect_min',
 'bilirubin_indirect_max',
 'ck_cpk_min',
 'ck_cpk_max',
 'ck_mb_min',
 'ck_mb_max',
 'ggt_min',
 'ggt_max',
 'ld_ldh_min',
 'ld_ldh_max',
 'so2_min',
 'so2_max',
 'po2_min',
 'po2_max']

# %%
#making a demographic dataframe
df_demographic = df_cohort_step3[["subject_id", "stay_id"] + Demographics].copy()
df_demographic.head()

# %%
# making only for static dataframe

df_static_labs = df_cohort_step3[["stay_id"] + static_features].copy()

df_static_labs.head()

# %%
#adding static suffix so we can distinguih them from time-series features later on
df_static_labs = df_static_labs.rename(columns={col: col + "_static" for col in static_features})

df_static_labs

# %%
#merge them according to stay_id
df_static_features = df_demographic.merge(df_static_labs, on="stay_id")

df_static_features

# %%
#exporting the dataframe to csv for step 4
df_static_features.to_csv("cohort_step_3.csv", index=False)


