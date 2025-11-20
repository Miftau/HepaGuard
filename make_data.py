import pandas as pd
import os

# --- Configuration ---
NHANES_DIR = 'nchs_data' # Directory containing the .xpt files
OUTPUT_DIR = 'hepaguard_datasets'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define target variable name
TARGET_COLUMN_NAME = 'LIVER_HIGH_RISK_PROFILE' # Changed name to reflect combination-based target
# Define thresholds for the combination (these are hypothetical and should be validated by a clinician)
GGT_THRESHOLD = 60  # Elevated GGT (example value)
PLT_THRESHOLD = 150 # Low Platelets (example value, units are 1000/uL, so 150 = 150,000)
GLU_HBA1C_HIGH = True # Use either high glucose OR high HbA1c as metabolic indicator
GLU_THRESHOLD = 140   # High Glucose (mg/dL)
HBA1C_THRESHOLD = 6.5 # High HbA1c (%)

# --- 1. Load Individual NHANES Files ---
def load_nhanes_file(file_name):
    """Load an NHANES XPT file into a DataFrame."""
    file_path = os.path.join(NHANES_DIR, file_name)
    try:
        df = pd.read_sas(file_path, format='xport')
        print(f"✅ Loaded {file_name}: Shape {df.shape}")
        # print(f"  - Columns: {list(df.columns)}") # Comment out column printing for cleaner output
        return df
    except FileNotFoundError:
        print(f"❌ File {file_name} not found in {NHANES_DIR}.")
        return None
    except Exception as e:
        print(f"❌ Error loading {file_name}: {e}")
        return None

print("Loading NHANES 2021-2023 data files...")

# Load the necessary NHANES files
demo_df = load_nhanes_file('DEMO_L.xpt') # Demographics (SEQN, Age, Gender, Ethnicity, Education)
bmx_df = load_nhanes_file('BMX_L.xpt')   # Body Measures (BMI, Waist Circumference)
smq_df = load_nhanes_file('SMQ_L.xpt')   # Smoking Questionnaire (Smoking status)
alq_df = load_nhanes_file('ALQ_L.xpt')   # Alcohol Questionnaire (Alcohol use)
diq_df = load_nhanes_file('DIQ_L.xpt')   # Diabetes Questionnaire (Diabetes status)
bpq_df = load_nhanes_file('BPQ_L.xpt')   # Blood Pressure Questionnaire (Hypertension status)
mcq_df = load_nhanes_file('MCQ_L.xpt')   # Medical Conditions Questionnaire (Family history)
# Note: HEPB_S_L.xpt was confirmed to have LBXHBS (Hep B Ab), not ALT/AST
hepa_df = load_nhanes_file('HEPA_L.xpt')   # Hepatitis A Antibody (LBXHA) - NOT GGT/Bili
# Note: L24GH_G.xpt was the suspected file for ALT/AST but is missing
cbc_df = load_nhanes_file('CBC_L.xpt')   # Complete Blood Count (Platelets - LBXPLTSI)
glu_df = load_nhanes_file('GLU_L.xpt')   # Glucose (LBXGLU)
ghb_df = load_nhanes_file('GHB_L.xpt')   # Glycohemoglobin (HbA1c - LBXGH)
hdl_df = load_nhanes_file('HDL_L.xpt')   # HDL Cholesterol (LBDHDD)
tchol_df = load_nhanes_file('TCHOL_L.xpt') # Total Cholesterol (LBXTC)
trigly_df = load_nhanes_file('TRIGLY_L.xpt') # Triglycerides (LBXTLG)

# Check if all required dataframes are loaded
# We need DEMO for SEQN, and at least some lab/risk factor data to define the target profile
critical_dfs = [demo_df, cbc_df, glu_df, ghb_df] # Need at least basic lab data for metabolic profile
other_dfs = [bmx_df, smq_df, alq_df, diq_df, bpq_df, mcq_df, hepa_df, hdl_df, tchol_df, trigly_df]

if demo_df is None:
    print("❌ Core demographic file (DEMO_L.xpt) is missing. Cannot proceed.")
    exit()

if all(df is None for df in critical_dfs):
    print("❌ All critical lab files (CBC_L, GLU_L, GHB_L) are missing. Cannot define a metabolic/liver risk profile.")
    exit()

# --- 2. Merge Dataframes on SEQN ---
print("\n--- Merging Datasets ---")

# Start with the core demographic data (contains SEQN)
merged_df = demo_df[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2']].copy()
print(f"Initial merged shape (Demo): {merged_df.shape}")

# Add Body Measures
if bmx_df is not None:
    merged_df = merged_df.merge(bmx_df[['SEQN', 'BMXBMI', 'BMXWAIST']], on='SEQN', how='left')
    print(f"After merging BMX: {merged_df.shape}")

# Add Smoking
if smq_df is not None:
    merged_df = merged_df.merge(smq_df[['SEQN', 'SMQ020', 'SMQ040']], on='SEQN', how='left')
    print(f"After merging SMQ: {merged_df.shape}")

# Add Alcohol
if alq_df is not None:
    merged_df = merged_df.merge(alq_df[['SEQN', 'ALQ130']], on='SEQN', how='left')
    print(f"After merging ALQ: {merged_df.shape}")

# Add Diabetes
if diq_df is not None:
    merged_df = merged_df.merge(diq_df[['SEQN', 'DIQ010']], on='SEQN', how='left')
    print(f"After merging DIQ: {merged_df.shape}")

# Add Blood Pressure
if bpq_df is not None:
    merged_df = merged_df.merge(bpq_df[['SEQN', 'BPQ020']], on='SEQN', how='left')
    print(f"After merging BPQ: {merged_df.shape}")

# Add Family History
if mcq_df is not None:
    merged_df = merged_df.merge(mcq_df[['SEQN', 'MCQ160E']], on='SEQN', how='left')
    print(f"After merging MCQ: {merged_df.shape}")

# --- NOTE: Skipping HEPA_L merge for GGT/BILI as they are not present ---
# if hepa_df is not None:
#     merged_df = merged_df.merge(hepa_df[['SEQN', 'LBXGGT', 'LBXBILI']], on='SEQN', how='left')
#     print(f"After merging HEPA_L (GGT/Bili): {merged_df.shape}")
# ELSE: Print confirmation that HEPA_L was loaded but no GGT/BILI found
if hepa_df is not None:
    print(f"INFO: HEPA_L.xpt loaded but no GGT/Bilirubin columns found for merging. Available: {list(hepa_df.columns)}")

# Add Platelets (CBC_L.xpt - LBXPLTSI) - AVAILABLE
if cbc_df is not None:
    merged_df = merged_df.merge(cbc_df[['SEQN', 'LBXPLTSI']], on='SEQN', how='left')
    print(f"After merging CBC (Platelets): {merged_df.shape}")

# Add Glucose (GLU_L.xpt - LBXGLU) - AVAILABLE
if glu_df is not None:
    merged_df = merged_df.merge(glu_df[['SEQN', 'LBXGLU']], on='SEQN', how='left')
    print(f"After merging GLU (Glucose): {merged_df.shape}")

# Add HbA1c (GHB_L.xpt - LBXGH) - AVAILABLE
if ghb_df is not None:
    merged_df = merged_df.merge(ghb_df[['SEQN', 'LBXGH']], on='SEQN', how='left')
    print(f"After merging GHB (HbA1c): {merged_df.shape}")

# Add HDL Cholesterol (HDL_L.xpt - LBDHDD) - AVAILABLE
if hdl_df is not None:
    merged_df = merged_df.merge(hdl_df[['SEQN', 'LBDHDD']], on='SEQN', how='left')
    print(f"After merging HDL (HDL Cholesterol): {merged_df.shape}")

# Add Total Cholesterol (TCHOL_L.xpt - LBXTC) - AVAILABLE
if tchol_df is not None:
    merged_df = merged_df.merge(tchol_df[['SEQN', 'LBXTC']], on='SEQN', how='left')
    print(f"After merging TCHOL (Total Cholesterol): {merged_df.shape}")

# Add Triglycerides (TRIGLY_L.xpt - LBXTLG) - AVAILABLE
if trigly_df is not None:
    merged_df = merged_df.merge(trigly_df[['SEQN', 'LBXTLG']], on='SEQN', how='left')
    print(f"After merging TRIGLY (Triglycerides): {merged_df.shape}")

# --- 3. Define Target Column (LIVER_HIGH_RISK_PROFILE) based on combination of available factors ---
print("\n--- Defining Target Variable (LIVER_HIGH_RISK_PROFILE) based on combination of factors ---")

# Define thresholds for the risk factors (these are illustrative and should ideally be based on clinical guidelines)
HIGH_GLUCOSE = 140 # mg/dL
HIGH_HBA1C = 6.5 # %
HIGH_TRIGS = 150 # mg/dL
LOW_HDL_MALE = 40 # mg/dL
LOW_HDL_FEMALE = 50 # mg/dL (example difference)
HIGH_CHOL = 240 # mg/dL
HIGH_BMI = 30 # Obese
IS_DIABETIC = 1 # Value for DIQ010 indicating 'Yes'
IS_SMOKER = 1    # Value for SMQ020 indicating 'Yes'
HEAVY_ALCOHOL = 1 # Value for ALQ130 indicating 'Yes' (needs confirmation of value meaning)
LOW_PLTS = 150 # x1000 cells/uL (Thrombocytopenia cut-off, approximate)

# Initialize the target column as 0 (Low Risk Profile)
merged_df[TARGET_COLUMN_NAME] = 0

# Create conditions for high risk profile based on available data
# Example Rule (MASH/NAFLD risk + structural indicator):
# High BMI OR (High Glucose OR High HbA1c OR Diabetic) AND (Low HDL OR High Trigs) AND NOT (Low Platelets)
# This is a simplified rule for demonstration. A real rule would be more complex and validated.
# Let's make a rule that captures metabolic dysfunction and adds other factors.
condition_metabolic_risk = (
    (merged_df.get('BMXBMI', 0) > HIGH_BMI) |
    (merged_df.get('LBXGLU', 0) > HIGH_GLUCOSE) |
    (merged_df.get('LBXGH', 0) > HIGH_HBA1C) |
    (merged_df.get('DIQ010', 0) == IS_DIABETIC)
)
condition_lipid_risk = (
    ((merged_df.get('RIAGENDR') == 1) & (merged_df.get('LBDHDD', 999) < LOW_HDL_MALE)) | # Male low HDL
    ((merged_df.get('RIAGENDR') == 2) & (merged_df.get('LBDHDD', 999) < LOW_HDL_FEMALE)) | # Female low HDL
    (merged_df.get('LBXTLG', 0) > HIGH_TRIGS)
)
condition_other_risk = (
    (merged_df.get('SMQ020', 0) == IS_SMOKER) |
    (merged_df.get('ALQ130', 0) == HEAVY_ALCOHOL) # Assuming 1 means heavy drinking - CHECK CODEBOOK
)
condition_structural_marker = (
    merged_df.get('LBXPLTSI', 1000) < LOW_PLTS # Low platelets
)

# Combine conditions into the final rule for the high-risk profile
# Example: Metabolic Risk AND (Lipid Risk OR Other Risk) OR Structural Marker
# This is a hypothetical combination. The exact logic needs clinical input.
rule_mask = (
    (condition_metabolic_risk & (condition_lipid_risk | condition_other_risk)) |
    condition_structural_marker
) & merged_df[['BMXBMI', 'LBXGLU', 'LBXGH', 'DIQ010', 'LBDHDD', 'LBXTLG', 'SMQ020', 'ALQ130', 'LBXPLTSI']].notna().all(axis=1) # Ensure all features used in the rule are not NaN

# Apply the rule mask to set the target variable
merged_df.loc[rule_mask, TARGET_COLUMN_NAME] = 1

print(f"✅ Created '{TARGET_COLUMN_NAME}' column based on a combination of metabolic, lipid, lifestyle, and structural markers.")
target_counts = merged_df[TARGET_COLUMN_NAME].value_counts().sort_index()
print(f"  - Distribution: {target_counts.to_dict()}")
if 1 in target_counts:
    pct_high_risk = (target_counts[1] / target_counts.sum()) * 100
    print(f"  - Percentage classified as High Risk Profile: {pct_high_risk:.2f}%")
else:
    print("  - No individuals met the criteria for the High Risk Profile in this merged dataset.")
    print("  - Consider adjusting thresholds or the combination rule.")

# --- 4. Prepare Clinical and General Datasets ---

# Define feature sets for Clinical and General models based on the merged data
# Clinical features: Primarily lab results and clinical measurements (available)
CLINICAL_FEATURES = [
    # 'LBXALT', 'LBXAST', # Not available
    # 'LBXGGT', 'LBXBILI', # Not available from HEPA_L.xpt as expected
    'LBXPLTSI', # From CBC - relates to structure
    'LBXGLU',   # From GLU - metabolic
    'LBXGH',    # From GHB - metabolic
    'LBDHDD',   # From HDL - lipid
    'LBXTC',    # From TCHOL - lipid
    'LBXTLG',   # From TRIGLY - lipid
    # Add Bilirubin if found in another file later, GGT if found later
]
# General/Lifestyle features: Demographics, Anthropometry, Behaviors, Conditions
GENERAL_FEATURES = [
    'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2',
    'BMXBMI', 'BMXWAIST', # Anthropometry
    'SMQ020', 'SMQ040',   # Smoking
    'ALQ130',             # Alcohol
    'DIQ010',             # Diabetes
    'BPQ020',             # Hypertension
    'MCQ160E'             # Family History (example, adjust if needed)
]

# Ensure all specified features are present in the merged dataframe
missing_clinical = [f for f in CLINICAL_FEATURES if f not in merged_df.columns]
missing_general = [f for f in GENERAL_FEATURES if f not in merged_df.columns]

if missing_clinical:
    print(f"⚠️  Missing Clinical Features in merged data: {missing_clinical}")
    # Remove missing features from the list to use
    CLINICAL_FEATURES = [f for f in CLINICAL_FEATURES if f in merged_df.columns]
    print(f"  - Using Clinical Features: {CLINICAL_FEATURES}")
if missing_general:
    print(f"⚠️  Missing General Features in merged data: {missing_general}")
    # Remove missing features from the list to use
    GENERAL_FEATURES = [f for f in GENERAL_FEATURES if f in merged_df.columns]
    print(f"  - Using General Features: {GENERAL_FEATURES}")

# Create Clinical Dataset (SEQN, Target, Clinical Features)
clinical_features_to_use = CLINICAL_FEATURES
# Select only the required columns for the clinical dataset
clinical_columns = ['SEQN', TARGET_COLUMN_NAME] + clinical_features_to_use
clinical_df = merged_df[clinical_columns].copy()
print(f"\n✅ Clinical Dataset: Shape {clinical_df.shape}, Features: {len(clinical_features_to_use)}")

# Create General Dataset (SEQN, Target, General Features)
general_features_to_use = GENERAL_FEATURES
# Select only the required columns for the general dataset
general_columns = ['SEQN', TARGET_COLUMN_NAME] + general_features_to_use
general_df = merged_df[general_columns].copy()
print(f"✅ General Dataset: Shape {general_df.shape}, Features: {len(general_features_to_use)}")

# --- 5. Save Datasets to CSV ---
clinical_output_path = os.path.join(OUTPUT_DIR, 'nhanes_liver_clinical.csv')
general_output_path = os.path.join(OUTPUT_DIR, 'nhanes_liver_general.csv')

clinical_df.to_csv(clinical_output_path, index=False)
general_df.to_csv(general_output_path, index=False)

print(f"\n✅ Datasets saved to:")
print(f"  - Clinical: {clinical_output_path}")
print(f"  - General: {general_output_path}")

print("\n--- HepaGuard Dataset Preparation Complete (Using Risk Profile Target) ---")
print("NOTE: The target variable 'LIVER_HIGH_RISK_PROFILE' is based on a combination of risk factors due to missing direct liver enzyme data (ALT/AST/GGT/Bili).")