import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import joblib # Import joblib for saving models and preprocessors

# --- Configuration ---
CLINICAL_DATA_FILE = 'hepaguard_datasets/nhanes_liver_clinical.csv' # Updated path
GENERAL_DATA_FILE = 'hepaguard_datasets/nhanes_liver_general.csv'   # Updated path
ID_COLUMN = 'SEQN'
TARGET_COLUMN_NAME = 'LIVER_HIGH_RISK_PROFILE' # Updated target name
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- 1. Load Data ---
print("Loading datasets...")
try:
    clinical_df = pd.read_csv(CLINICAL_DATA_FILE)
    general_df = pd.read_csv(GENERAL_DATA_FILE)
    print(f"✅ Clinical data loaded: Shape {clinical_df.shape}")
    print(f"✅ General data loaded: Shape {general_df.shape}")
    print(f"  - Clinical columns: {list(clinical_df.columns)}")
    print(f"  - General columns: {list(general_df.columns)}")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Please ensure 'nhanes_liver_clinical.csv' and 'nhanes_liver_general.csv' are in the script directory.")
    exit()

# --- 2. Verify Target Column Exists ---
print("\n--- Verifying Target Column ---")
if TARGET_COLUMN_NAME not in clinical_df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN_NAME}' not found in {CLINICAL_DATA_FILE}")
else:
    print(f"✅ Target column '{TARGET_COLUMN_NAME}' found in Clinical data.")
    print(f"  - Distribution: {clinical_df[TARGET_COLUMN_NAME].value_counts().sort_index()}")
    print(f"  - Distribution (%): {(clinical_df[TARGET_COLUMN_NAME].value_counts(normalize=True) * 100).round(2).sort_index()}")

if TARGET_COLUMN_NAME not in general_df.columns:
     raise ValueError(f"Target column '{TARGET_COLUMN_NAME}' not found in {GENERAL_DATA_FILE}")
else:
    print(f"✅ Target column '{TARGET_COLUMN_NAME}' found in General data.")
    print(f"  - Distribution: {general_df[TARGET_COLUMN_NAME].value_counts().sort_index()}")
    print(f"  - Distribution (%): {(general_df[TARGET_COLUMN_NAME].value_counts(normalize=True) * 100).round(2).sort_index()}")

# --- 3. Data Exploration and Visualization (Optional - Can be commented out) ---
# The explore_and_visualize function and its calls are kept for completeness but can be skipped for faster execution
def explore_and_visualize(df, name, target_col, feature_cols_subset=None):
    """Performs basic exploration and visualization for a given dataframe."""
    print(f"\n--- Data Exploration: {name} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    if target_col in df.columns:
        print(f"Target Distribution ({target_col}):")
        print(df[target_col].value_counts())
        print(f"Target Distribution (%):\n{df[target_col].value_counts(normalize=True) * 100}")
    else:
        print(f"⚠️  Target column '{target_col}' not found in {name} dataset.")

    # --- Visualization ---
    # 1. Missing Values Heatmap (for the subset of features used in modeling, plus target)
    cols_to_plot_missing = [target_col] + (feature_cols_subset if feature_cols_subset else df.columns.tolist())
    cols_present = [c for c in cols_to_plot_missing if c in df.columns]
    if cols_present:
        plt.figure(figsize=(max(len(cols_present)*0.8, 8), 6))
        sns.heatmap(df[cols_present].isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title(f'Missing Values Heatmap - {name} (Subset)')
        plt.tight_layout()
        plt.savefig(f'hepaguard_{name.lower()}_missing_values_heatmap_subset.png')
        plt.show()

    # 2. Target Distribution Bar Plot (if target exists)
    if target_col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=target_col)
        plt.title(f'Target Distribution - {name}')
        plt.ylabel('Count')
        plt.xlabel(target_col)
        plt.tight_layout()
        plt.savefig(f'hepaguard_{name.lower()}_target_dist.png')
        plt.show()

        # Plotly Interactive Target Distribution
        fig = px.histogram(df, x=target_col, title=f'Interactive Target Distribution - {name}', color=target_col)
        fig.show()

    # 3. Feature Distribution (Numerical Subset) - Example: specified features
    if feature_cols_subset:
        numerical_cols_subset = [c for c in feature_cols_subset if c in df.select_dtypes(include=[np.number]).columns]
        if numerical_cols_subset:
            # Remove target if it's numerical for plotting purposes
            if target_col in numerical_cols_subset:
                 numerical_cols_subset.remove(target_col)
            cols_to_plot = numerical_cols_subset
            if cols_to_plot:
                fig, axes = plt.subplots(2, (len(cols_to_plot) + 1) // 2, figsize=(15, 10))
                if len(cols_to_plot) == 1:
                    axes = [axes] # Make iterable if only one subplot
                axes = axes.ravel()
                for i, col in enumerate(cols_to_plot):
                    if i < len(axes):
                        axes[i].hist(df[col].dropna(), bins=30, edgecolor='black')
                        axes[i].set_title(f'Distribution of {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequency')
                plt.tight_layout()
                plt.savefig(f'hepaguard_{name.lower()}_feature_dist_subset.png')
                plt.show()

                # Plotly Interactive Feature Distributions for subset
                for col in cols_to_plot:
                     fig = px.histogram(df, x=col, title=f'Interactive Distribution of {col} - {name}', marginal='box')
                     fig.show()

    # 4. Correlation Heatmap (Numerical Subset features only, including target if numerical)
    if feature_cols_subset:
        numerical_cols_subset_corr = [c for c in feature_cols_subset if c in df.select_dtypes(include=[np.number]).columns]
        if target_col in df.select_dtypes(include=[np.number]).columns:
            numerical_cols_subset_corr.append(target_col) # Include target if it's numerical

        if len(numerical_cols_subset_corr) > 1:
            plt.figure(figsize=(max(len(numerical_cols_subset_corr)*0.8, 8), max(len(numerical_cols_subset_corr)*0.5, 6)))
            correlation_matrix = df[numerical_cols_subset_corr].corr()
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
            plt.title(f'Correlation Heatmap (Subset Features + Target) - {name}')
            plt.tight_layout()
            plt.savefig(f'hepaguard_{name.lower()}_correlation_heatmap_subset.png')
            plt.show()
        else:
            print(f"⚠️  Not enough numerical features (subset + target) to plot correlation heatmap for {name}.")

# Define feature lists based on the actual columns in the loaded CSVs
CLINICAL_FEATURES = ['LBXPLTSI', 'LBXGLU', 'LBXGH', 'LBDHDD', 'LBXTC', 'LBXTLG'] # Exclude SEQN and LIVER_HIGH_RISK_PROFILE
GENERAL_FEATURES = ['RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'BMXBMI', 'BMXWAIST', 'SMQ020', 'SMQ040', 'ALQ130', 'DIQ010', 'BPQ020', 'MCQ160E'] # Exclude SEQN and LIVER_HIGH_RISK_PROFILE

# Explore Clinical dataset using its specific features
explore_and_visualize(clinical_df, "Clinical", TARGET_COLUMN_NAME, CLINICAL_FEATURES)

# Explore General dataset using its specific features
explore_and_visualize(general_df, "General", TARGET_COLUMN_NAME, GENERAL_FEATURES)


# --- 4. Prepare Data for Modeling ---
def prepare_data(df, feature_cols, target_col, id_col=None):
    """Separate specified features (X) and target (y), handle missing values, encode categories."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset after definition step.")

    # Select only the specified features for X
    X = df[feature_cols].copy()
    # Optionally, remove the ID column if it's present in the feature list
    if id_col and id_col in X.columns:
        X = X.drop(columns=[id_col])
    y = df[target_col] # Target is taken from the full dataframe

    # Identify categorical and numerical columns *within the specified feature set*
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    print(f"  - Specified Features: {len(feature_cols)}")
    print(f"    - Numerical (from spec): {len(numerical_cols)}, Categorical (from spec): {len(categorical_cols)}")

    # Preprocessing pipeline for numerical data
    # Use median for imputation, standardize for models like Logistic Regression, SVM
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) # Standardization is important for Logistic Regression, SVM, Neural Nets, XGBoost
    ])

    # Preprocessing pipeline for categorical data (if any exist in the specified set)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # Fill missing categories
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # One-hot encode, drop first to avoid multicollinearity
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return X, y, preprocessor, numerical_cols, categorical_cols

# Prepare datasets using the specific feature lists found in the CSVs
print("\n--- Preparing Clinical Data for Modeling (using features from CSV) ---")
X_clin_csv, y_clin_csv, preprocessor_clin, num_cols_clin_csv, cat_cols_clin_csv = prepare_data(
    clinical_df, CLINICAL_FEATURES, TARGET_COLUMN_NAME, ID_COLUMN
)

print("\n--- Preparing General Data for Modeling (using features from CSV) ---")
X_gen_csv, y_gen_csv, preprocessor_gen, num_cols_gen_csv, cat_cols_gen_csv = prepare_data(
    general_df, GENERAL_FEATURES, TARGET_COLUMN_NAME, ID_COLUMN # Note: TARGET_COLUMN_NAME is not in GENERAL_FEATURES, but y is taken separately
)

# --- 5. Prepare Combined Data for Modeling ---
# Merge on the unique identifier (SEQN) using the full dataframes before selecting features
print("\n--- Preparing Combined Data for Modeling ---")
# Use 'inner' join to ensure both clinical and general features are available for each participant
# This requires both SEQN values to match between the two dataframes.
# It's crucial that the LIVER_HIGH_RISK_PROFILE is consistent between the two files for the same SEQN.
# If it's derived from different rules in each file, this could be problematic. Assuming it's the same rule applied to the merged dataset *before* splitting.
# For this script, we assume the target in the clinical file is the ground truth to use.
clinical_df_indexed = clinical_df.set_index(ID_COLUMN)[[TARGET_COLUMN_NAME] + CLINICAL_FEATURES]
general_df_indexed = general_df.set_index(ID_COLUMN)[GENERAL_FEATURES]

# Perform the inner join
combined_df_indexed = clinical_df_indexed.join(general_df_indexed, how='inner') # Inner join keeps only where both sides have data

# Separate features and target for the combined dataset
y_combined_csv = combined_df_indexed[TARGET_COLUMN_NAME]
X_combined_csv = combined_df_indexed.drop(columns=[TARGET_COLUMN_NAME])

print(f"✅ Combined data shape (inner join on {ID_COLUMN}): {combined_df_indexed.shape}")
print(f"  - Combined Features (Clinical + General): {X_combined_csv.shape[1]}")
print(f"  - Combined Target: Present for {y_combined_csv.shape[0]} participants")

# Identify categorical and numerical columns for combined data (subset)
categorical_cols_combined_csv = X_combined_csv.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols_combined_csv = X_combined_csv.select_dtypes(include=['number']).columns.tolist()
print(f"  - Combined Features (from CSV join): {X_combined_csv.shape[1]}")
print(f"    - Numerical (from CSV join): {len(numerical_cols_combined_csv)}, Categorical (from CSV join): {len(categorical_cols_combined_csv)}")

# Preprocessing pipeline for combined data (subset from joined CSVs)
numerical_transformer_combined_csv = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer_combined_csv = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])
preprocessor_combined_csv = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_combined_csv, numerical_cols_combined_csv),
        ('cat', categorical_transformer_combined_csv, categorical_cols_combined_csv)
    ]
)


# --- 6. Define Models to Train ---
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
    'XGBoost': xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
    'Support Vector Machine': SVC(random_state=RANDOM_STATE, probability=True) # Enable probability for ROC AUC
    # Add Neural Network example if needed (using sklearn's MLPClassifier)
    # 'Neural Network': MLPClassifier(random_state=RANDOM_STATE, max_iter=500)
}

# --- 7. Train and Evaluate Models ---
def train_and_evaluate(X, y, preprocessor, model_dict, dataset_name):
    """Train multiple models and evaluate their performance."""
    results = []
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in model_dict.items():
        print(f"\n--- Training {name} on {dataset_name} Data (Features: {X.shape[1]}) ---")
        # Create pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1] # Probability of positive class

        # Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0) # Handle cases with no predicted positives
        recall = recall_score(y_test, y_pred, zero_division=0)      # Handle cases with no actual positives
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Store results
        results.append({
            'Model': name,
            'Dataset': dataset_name,
            'Features_Count': X.shape[1], # Number of features used for this dataset/model combo
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc_roc
        })

        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")

        # Classification Report
        print(f"\nClassification Report for {name} on {dataset_name}:")
        print(classification_report(y_test, y_pred))

        # --- Visualization for Individual Model (Optional) ---
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name} ({dataset_name})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'cm_{name.lower()}_{dataset_name.lower()}.png')
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{name} (AUC = {auc_roc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No Skill')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name} ({dataset_name})')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'roc_{name.lower()}_{dataset_name.lower()}.png')
        plt.show()

    return results

# Train and evaluate models for Clinical data (using features from CSV)
results_clinical = train_and_evaluate(X_clin_csv, y_clin_csv, preprocessor_clin, models, "Clinical (CSV Features)")

# Train and evaluate models for General data (using features from CSV)
results_general = train_and_evaluate(X_gen_csv, y_gen_csv, preprocessor_gen, models, "General (CSV Features)")

# Train and evaluate models for Combined data (using features from joined CSVs)
results_combined = train_and_evaluate(X_combined_csv, y_combined_csv, preprocessor_combined_csv, models, "Combined (CSV Features)")

# Combine results
all_results = results_clinical + results_general + results_combined
results_df = pd.DataFrame(all_results)

# --- 8. Compare Models ---
print("\n--- Model Comparison ---")
print(results_df.round(4))

# --- Visualization: Model Comparison (Optional) ---
# 1. Bar plot for AUC-ROC comparison
fig = px.bar(results_df, x='Model', y='AUC-ROC', color='Dataset', barmode='group',
             title='Model Comparison: AUC-ROC Score (Using CSV Features)',
             labels={'AUC-ROC': 'AUC-ROC Score', 'Model': 'Model Type', 'Dataset': 'Dataset Type'})
fig.show()

# 2. Bar plot for F1-Score comparison
fig = px.bar(results_df, x='Model', y='F1-Score', color='Dataset', barmode='group',
             title='Model Comparison: F1-Score (Using CSV Features)',
             labels={'F1-Score': 'F1-Score', 'Model': 'Model Type', 'Dataset': 'Dataset Type'})
fig.show()

# 3. Radar Chart for multiple metrics (example for Clinical dataset)
clinical_results = results_df[results_df['Dataset'] == 'Clinical (CSV Features)']
fig = go.Figure(data=go.Scatterpolar(
    r=clinical_results[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].values.T,
    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    fill='toself',
    name='Clinical Models (CSV)'
))
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    title=f"Model Performance Radar - Clinical Dataset (Features from CSV)",
    showlegend=True
)
fig.show()

# 4. Radar Chart for multiple metrics (example for General dataset)
general_results = results_df[results_df['Dataset'] == 'General (CSV Features)']
fig = go.Figure(data=go.Scatterpolar(
    r=general_results[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].values.T,
    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    fill='toself',
    name='General Models (CSV)'
))
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    title=f"Model Performance Radar - General Dataset (Features from CSV)",
    showlegend=True
)
fig.show()

# 5. Radar Chart for multiple metrics (example for Combined dataset)
combined_results = results_df[results_df['Dataset'] == 'Combined (CSV Features)']
fig = go.Figure(data=go.Scatterpolar(
    r=combined_results[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].values.T,
    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    fill='toself',
    name='Combined Models (CSV)'
))
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    title=f"Model Performance Radar - Combined Dataset (Features from CSV)",
    showlegend=True
)
fig.show()

# --- 9. Save Results ---
results_df.to_csv('hepaguard_model_comparison_results_csv_features.csv', index=False)
print(f"\n✅ Model comparison results (using features from CSVs) saved to 'hepaguard_model_comparison_results_csv_features.csv'")

# --- 10. Save Best Random Forest Models and Preprocessors ---
print("\n--- Saving Best Random Forest Models and Preprocessors ---")

# Identify the best Random Forest model for each dataset based on AUC-ROC
# Filter results for Random Forest only
rf_results = results_df[results_df['Model'] == 'Random Forest'].copy()
print("Random Forest Results:")
print(rf_results)

if not rf_results.empty:
    # Find best performer for each dataset
    for dataset_type in ['Clinical (CSV Features)', 'General (CSV Features)', 'Combined (CSV Features)']:
        dataset_results = rf_results[rf_results['Dataset'] == dataset_type]
        if not dataset_results.empty:
            best_idx = dataset_results['AUC-ROC'].idxmax()
            best_rf_row = dataset_results.loc[best_idx]
            best_model_name_on_dataset = best_rf_row['Model']
            best_auc_on_dataset = best_rf_row['AUC-ROC']
            print(f"\nBest Random Forest for {dataset_type}: AUC-ROC = {best_auc_on_dataset:.4f}")

            # Determine which preprocessor and data were used for this specific model/dataset combo
            # This requires keeping track of the *fitted* preprocessors. We need to refit the preprocessor on the *full* training set for saving.
            # The `preprocessor_clin`, `preprocessor_gen`, `preprocessor_combined_csv` defined earlier were fitted during the `train_and_evaluate` function's split.
            # To save the final model ready for prediction on *new* data, we should fit the preprocessor on the *entire* X_train and the model on the preprocessed X_train.
            # However, `train_and_evaluate` already created pipelines. We can extract the fitted parts if needed, but it's cleaner to fit a final pipeline here.
            # Let's refit the preprocessor and model on the full dataset (X, y) for this specific type to get a final, deployable object.

            # Select the correct raw X and y based on dataset_type
            if dataset_type.startswith("Clinical"):
                X_to_fit = X_clin_csv
                y_to_fit = y_clin_csv
                preprocessor_to_fit = preprocessor_clin
                model_to_save = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100) # Use the same base model
            elif dataset_type.startswith("General"):
                X_to_fit = X_gen_csv
                y_to_fit = y_gen_csv
                preprocessor_to_fit = preprocessor_gen
                model_to_save = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
            elif dataset_type.startswith("Combined"):
                X_to_fit = X_combined_csv
                y_to_fit = y_combined_csv
                preprocessor_to_fit = preprocessor_combined_csv
                model_to_save = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
            else:
                print(f"Warning: Unrecognized dataset type for saving: {dataset_type}")
                continue # Skip saving for this unrecognized type

            # Fit the preprocessor on the full dataset
            print(f"  - Fitting preprocessor for {dataset_type}...")
            X_preprocessed_for_saving = preprocessor_to_fit.fit_transform(X_to_fit)
            print(f"    - Preprocessor fitted. Transformed shape: {X_preprocessed_for_saving.shape}")

            # Fit the model on the preprocessed data
            print(f"  - Fitting Random Forest model for {dataset_type}...")
            model_to_save.fit(X_preprocessed_for_saving, y_to_fit)
            print(f"    - Model fitted.")

            # --- Save the fitted model and preprocessor ---
            # Create filenames based on the dataset type
            model_filename = f"hepaguard_rf_model_{dataset_type.split()[0].lower()}.pkl" # e.g., hepaguard_rf_model_clinical.pkl
            preprocessor_filename = f"hepaguard_preprocessor_{dataset_type.split()[0].lower()}.pkl" # e.g., hepaguard_preprocessor_clinical.pkl

            # Save the model
            joblib.dump(model_to_save, model_filename)
            print(f"    - Saved Model: {model_filename}")

            # Save the preprocessor (the fitted ColumnTransformer)
            joblib.dump(preprocessor_to_fit, preprocessor_filename)
            print(f"    - Saved Preprocessor: {preprocessor_filename}")
        else:
            print(f"No Random Forest results found for {dataset_type}. Skipping save.")
else:
    print("❌ No Random Forest models were trained (results are empty). Cannot save.")

# --- 11. Optional: Hyperparameter Tuning for Top Performers ---
print("\n--- Optional: Hyperparameter Tuning for Top Performer (Random Forest - Clinical CSV Features) ---")
# Use the best performing model type and dataset as a starting point from your results
# For example, let's tune Random Forest on Clinical data (using features from CSV)
rf_clin_csv = RandomForestClassifier(random_state=RANDOM_STATE)
param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}
pipeline_rf_clin_csv = Pipeline(steps=[('preprocessor', preprocessor_clin), ('classifier', rf_clin_csv)])

# --- CRITICAL FIX: Define 'cv' variable for GridSearchCV ---
cv_for_tuning = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

grid_search_rf_clin_csv = GridSearchCV(
    pipeline_rf_clin_csv,
    param_grid_rf,
    cv=cv_for_tuning, # Use the defined cv object
    scoring='roc_auc',
    n_jobs=-1, # Use all available cores
    verbose=1  # Print progress
)

# Split data for tuning (using the clinical data prepared earlier)
X_train_clin_csv, X_test_clin_csv, y_train_clin_csv, y_test_clin_csv = train_test_split(
    X_clin_csv, y_clin_csv, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clin_csv
)

# Fit the grid search
grid_search_rf_clin_csv.fit(X_train_clin_csv, y_train_clin_csv)

# Get the best estimator and its score
best_rf_clin_csv = grid_search_rf_clin_csv.best_estimator_
best_score_rf_clin_csv = grid_search_rf_clin_csv.best_score_

print(f"Best Random Forest (Clinical CSV Features) parameters: {grid_search_rf_clin_csv.best_params_}")
print(f"Best CV AUC-ROC score: {best_score_rf_clin_csv:.4f}")

# Evaluate the tuned model on the held-out test set
y_pred_best_rf_clin_csv = best_rf_clin_csv.predict(X_test_clin_csv)
y_pred_proba_best_rf_clin_csv = best_rf_clin_csv.predict_proba(X_test_clin_csv)[:, 1] # Probability of positive class
auc_best_rf_clin_csv = roc_auc_score(y_test_clin_csv, y_pred_proba_best_rf_clin_csv)
print(f"Tuned Random Forest (Clinical CSV Features) Test AUC-ROC: {auc_best_rf_clin_csv:.4f}")

# --- End ---
print("\n--- HepaGuard Model Training and Evaluation (Features from CSVs) Complete ---")
print("Check the generated plots and the 'hepaguard_model_comparison_results_csv_features.csv' file.")
print("The optional hyperparameter tuning results are printed above.")
print("\n--- Models and Preprocessors Saved ---")
print("Look for the following .pkl files in your script directory:")
print("  - hepaguard_rf_model_clinical.pkl")
print("  - hepaguard_preprocessor_clinical.pkl")
print("  - hepaguard_rf_model_general.pkl")
print("  - hepaguard_preprocessor_general.pkl")
print("  - hepaguard_rf_model_combined.pkl")
print("  - hepaguard_preprocessor_combined.pkl")