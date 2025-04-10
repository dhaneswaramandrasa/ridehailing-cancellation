import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO # BytesIO needed for uploaded file
import hashlib # To help with caching based on file content
import warnings

# --- Sklearn & Imblearn Imports ---
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline # Use sklearn Pipeline for transformers
from imblearn.pipeline import Pipeline as ImbPipeline # Use imblearn Pipeline for SMOTE + Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, accuracy_score
from imblearn.over_sampling import SMOTE

# --- Statsmodels Import ---
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# --- Configuration ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5) # Adjust default plot size for dashboard

# --- Expected Schema ---
EXPECTED_COLUMNS = [
    'ride_id', 'user_id', 'driver_id', 'start_time', 'end_time',
    'pickup_location', 'dropoff_location', 'fare', 'distance_km',
    'rating', 'cancelled'
]

# --- Caching Functions ---

def get_file_hash(file_obj):
    if file_obj is None: return "default_sample_data"
    file_obj.seek(0); hasher = hashlib.md5(); hasher.update(file_obj.read()); file_obj.seek(0)
    return hasher.hexdigest()

@st.cache_data
def load_and_process_data(file_input_hash, uploaded_file_obj):
    """Loads data and performs extensive feature engineering."""
    df = None
    data_source = "Sample Data"
    try:
        if uploaded_file_obj is not None:
            st.info("Processing uploaded file...")
            df = pd.read_csv(uploaded_file_obj)
            data_source = f"Uploaded File: `{uploaded_file_obj.name}`"
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            if missing_cols:
                st.error(f"Uploaded file missing columns: {', '.join(missing_cols)}.")
                return None, None, None, None, data_source
        else:
            # Default Sample Data
            csv_data = """ride_id,user_id,driver_id,start_time,end_time,pickup_location,dropoff_location,fare,distance_km,rating,cancelled
R0001,U1,D1,2025-01-01 08:00:00,2025-01-01 08:30:00,LocA,LocB,5,1,5,0;R0002,U2,D2,2025-01-01 09:00:00,2025-01-01 09:30:00,LocB,LocC,5.5,1.2,4,0;R0003,U3,D3,2025-01-01 10:00:00,2025-01-01 10:30:00,LocC,LocD,6,1.4,3,1;R0004,U4,D4,2025-01-01 11:00:00,2025-01-01 11:30:00,LocD,LocA,6.5,1.6,2,0;R0005,U5,D1,2025-01-01 12:00:00,2025-01-01 12:30:00,LocA,LocB,7,1.8,1,0;R0006,U1,D2,2025-01-01 13:00:00,2025-01-01 13:30:00,LocB,LocC,7.5,2,5,0;R0007,U2,D3,2025-01-01 14:00:00,2025-01-01 14:30:00,LocC,LocD,8,2.2,4,0;R0008,U3,D4,2025-01-01 15:00:00,2025-01-01 15:30:00,LocD,LocA,8.5,2.4,3,1;R0009,U4,D1,2025-01-01 16:00:00,2025-01-01 16:30:00,LocA,LocB,9,2.6,2,0;R0010,U5,D2,2025-01-01 17:00:00,2025-01-01 17:30:00,LocB,LocC,9.5,2.8,1,0;R0011,U1,D3,2025-01-01 18:00:00,2025-01-01 18:30:00,LocC,LocD,10,3,5,0;R0012,U2,D4,2025-01-01 19:00:00,2025-01-01 19:30:00,LocD,LocA,10.5,3.2,4,0;R0013,U3,D1,2025-01-01 20:00:00,2025-01-01 20:30:00,LocA,LocB,11,3.4,3,1;R0014,U4,D2,2025-01-01 21:00:00,2025-01-01 21:30:00,LocB,LocC,11.5,3.6,2,0;R0015,U5,D3,2025-01-01 22:00:00,2025-01-01 22:30:00,LocC,LocD,12,3.8,1,0;R0016,U1,D4,2025-01-01 23:00:00,2025-01-01 23:30:00,LocD,LocA,12.5,4,5,0;R0017,U2,D1,2025-01-02 00:00:00,2025-01-02 00:30:00,LocA,LocB,13,4.2,4,0;R0018,U3,D2,2025-01-02 01:00:00,2025-01-02 01:30:00,LocB,LocC,13.5,4.4,3,1;R0019,U4,D3,2025-01-02 02:00:00,2025-01-02 02:30:00,LocC,LocD,14,4.6,2,0;R0020,U5,D4,2025-01-02 03:00:00,2025-01-02 03:30:00,LocD,LocA,14.5,4.8,1,0"""
            df = pd.read_csv(StringIO(csv_data.replace(';', '\n'))) # Ensure newlines

        # --- Cleaning and Basic Feature Engineering ---
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
        for col in ['fare', 'distance_km', 'rating', 'cancelled']:
             if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['start_time', 'end_time', 'fare', 'distance_km', 'cancelled'], inplace=True)
        df['cancelled'] = df['cancelled'].astype(int)

        time_conversion_success = pd.api.types.is_datetime64_any_dtype(df['start_time']) and pd.api.types.is_datetime64_any_dtype(df['end_time'])

        if time_conversion_success:
            df['trip_duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
            df['start_hour'] = df['start_time'].dt.hour
            df['start_dayofweek'] = df['start_time'].dt.dayofweek
            df['start_date'] = df['start_time'].dt.date
            epsilon = 1e-6
            df['avg_speed_kmh'] = df['distance_km'] / ((df['trip_duration_minutes'].fillna(0) + epsilon) / 60)
            df.loc[df['distance_km'] == 0, 'avg_speed_kmh'] = 0
            df.loc[df['trip_duration_minutes'].fillna(0) <= 0, 'avg_speed_kmh'] = 0
        else:
             # Add placeholder columns if time conversion failed, needed for model features
             df['trip_duration_minutes'] = np.nan
             df['start_hour'] = np.nan
             df['start_dayofweek'] = np.nan
             df['start_date'] = pd.NaT
             df['avg_speed_kmh'] = np.nan

        df = df.sort_values(by='start_time').reset_index(drop=True)

        # --- Advanced Feature Engineering (from Notebook) ---
        st.write("Calculating advanced features...")
        metrics_to_roll = ['fare', 'distance_km', 'trip_duration_minutes', 'cancelled', 'rating']
        window_size = 3

        # User Rolling Features
        df = df.sort_values(by=['user_id', 'start_time'])
        for metric in metrics_to_roll:
            if metric in df.columns:
                df[f'user_prev_{metric}'] = df.groupby('user_id')[metric].shift(1)
                df[f'user_rolling_mean_{window_size}_{metric}'] = df.groupby('user_id')[metric].shift(1).rolling(window=window_size, min_periods=1).mean()
                if metric not in ['cancelled', 'rating']:
                     df[f'user_rolling_sum_{window_size}_{metric}'] = df.groupby('user_id')[metric].shift(1).rolling(window=window_size, min_periods=1).sum()
        df[f'user_rolling_cancel_rate_{window_size}'] = df.groupby('user_id')['cancelled'].shift(1).rolling(window=window_size, min_periods=1).mean()

        # Driver Rolling Features
        df = df.sort_values(by=['driver_id', 'start_time'])
        for metric in metrics_to_roll:
             if metric in df.columns:
                df[f'driver_prev_{metric}'] = df.groupby('driver_id')[metric].shift(1)
                df[f'driver_rolling_mean_{window_size}_{metric}'] = df.groupby('driver_id')[metric].shift(1).rolling(window=window_size, min_periods=1).mean()
                if metric not in ['cancelled', 'rating']:
                    df[f'driver_rolling_sum_{window_size}_{metric}'] = df.groupby('driver_id')[metric].shift(1).rolling(window=window_size, min_periods=1).sum()
        df[f'driver_rolling_cancel_rate_{window_size}'] = df.groupby('driver_id')['cancelled'].shift(1).rolling(window=window_size, min_periods=1).mean()

        # RFM-like Features
        df = df.sort_values(by=['user_id', 'start_time'])
        df['user_ride_count_before'] = df.groupby('user_id').cumcount()
        if time_conversion_success:
            df['user_prev_start_time'] = df.groupby('user_id')['start_time'].shift(1)
            df['user_days_since_last_ride'] = (df['start_time'] - df['user_prev_start_time']).dt.total_seconds() / (60*60*24)
            df = df.drop(columns=['user_prev_start_time'], errors='ignore')
        else: df['user_days_since_last_ride'] = np.nan
        df['user_total_fare_before'] = df.groupby('user_id')['fare'].shift(1).expanding().sum()
        df['user_avg_fare_before'] = df.groupby('user_id')['fare'].shift(1).expanding().mean()
        df['user_total_cancelled_before'] = df.groupby('user_id')['cancelled'].shift(1).expanding().sum()
        df['user_cancel_rate_before'] = (df['user_total_cancelled_before'] / df['user_ride_count_before'])
        df = df.drop(columns=['user_total_cancelled_before'], errors='ignore')

        # NaN Handling for Engineered Features
        fill_zero_cols = [col for col in df.columns if 'prev_' in col or 'rolling_' in col or '_before' in col]
        if 'user_days_since_last_ride' in fill_zero_cols: fill_zero_cols.remove('user_days_since_last_ride')
        for col in fill_zero_cols:
            if col in df.columns: df[col] = df[col].fillna(0)
        if 'user_days_since_last_ride' in df.columns: df['user_days_since_last_ride'] = df['user_days_since_last_ride'].fillna(-1)
        if 'user_cancel_rate_before' in df.columns: df['user_cancel_rate_before'] = df['user_cancel_rate_before'].fillna(0)
        if 'avg_speed_kmh' in df.columns: df['avg_speed_kmh'] = df['avg_speed_kmh'].fillna(df['avg_speed_kmh'].median() if pd.notna(df['avg_speed_kmh'].median()) else 0)

        # Final Sort
        df = df.sort_values(by='start_time').reset_index(drop=True)
        st.write("Advanced feature calculation complete.")

        # --- Aggregations (using the feature-rich df) ---
        user_summary = df.groupby('user_id').agg(
            total_rides=('ride_id', 'count'), total_spending=('fare', 'sum'),
            avg_rating=('rating', 'mean'), last_ride_time=('start_time', 'max')
        ).reset_index()
        if not user_summary.empty and 'last_ride_time' in user_summary.columns and time_conversion_success:
             max_time = df['start_time'].max()
             if pd.notna(max_time): user_summary['recency_hours'] = (max_time - user_summary['last_ride_time']).dt.total_seconds() / 3600
             else: user_summary['recency_hours'] = np.nan
        else: user_summary['recency_hours'] = np.nan

        driver_summary = df.groupby('driver_id').agg(
            total_rides=('ride_id', 'count'), avg_rating_received=('rating', 'mean'),
            total_cancelled=('cancelled', lambda x: x.eq(1).sum())
        ).reset_index()
        if not driver_summary.empty and 'total_rides' in driver_summary.columns and driver_summary['total_rides'].sum() > 0 :
            driver_summary['cancellation_rate'] = (driver_summary['total_cancelled'] / driver_summary['total_rides']).fillna(0)
        else: driver_summary['cancellation_rate'] = np.nan

        hourly_demand = pd.Series(dtype=float)
        if time_conversion_success and not df.empty:
            df_ts = df.set_index('start_time')
            if pd.api.types.is_datetime64_any_dtype(df_ts.index):
                 hourly_demand = df_ts['ride_id'].resample('H').count().fillna(0)

        st.success(f"Data loaded and processed successfully from: {data_source}")
        return df, user_summary, driver_summary, hourly_demand, data_source

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None, None, None, None, data_source

def get_df_hash(dataframe):
    if dataframe is None: return "no_dataframe"
    return hashlib.sha256(pd.util.hash_pandas_object(dataframe, index=True).values).hexdigest()

# --- Model Training Function (Using Notebook Logic) ---
@st.cache_data
def train_evaluate_cv_model(df_hash, _df_processed):
    """Performs CV using notebook logic, returns avg metrics, and trains final model."""
    if _df_processed is None or _df_processed.empty:
        st.warning("No data available for model training.")
        return None, None, None

    df_model = _df_processed.copy()
    target = 'cancelled'

    # --- Feature Selection (Matching Notebook Part 3.1) ---
    potential_features = [col for col in df_model.columns if col not in [
        'ride_id', 'user_id', 'driver_id', 'cancelled',
        'start_time', 'end_time', 'start_date',
        'rating', # Rating is post-ride, leaky
        'trip_duration_minutes', 'avg_speed_kmh', # Leaky if calculated using end_time
        'user_prev_start_time', 'start_dayname' # Redundant or handled elsewhere
    ]]
    features = [f for f in potential_features if f in df_model.columns] # Ensure they exist

    if not features or target not in df_model.columns:
        st.error("Required features or target missing for model training.")
        return None, None, None

    X = df_model[features]
    y = df_model[target]

    if y.nunique() < 2:
        st.warning("Target variable has only one class. Cannot train/evaluate classification model.")
        return None, None, None

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- Preprocessing Pipeline (Matching Notebook Part 3.2) ---
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)], remainder='passthrough')

    # --- Model and SMOTE Setup (Matching Notebook Part 3.3 - Corrected k) ---
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    n_minority = y.value_counts().min()

    # Determine CV Splits first
    n_splits = min(n_minority, 4) # Use max 3 folds, limited by minority class

    smote_k_neighbors = min(n_minority - 2, 2) # Aim for 2, but ensure it's <= n_minority-2
    smote_k_neighbors = max(1, smote_k_neighbors) # Ensure k is at least 1
    smote_used = False
    pipeline_cv = None

    if n_splits < 2:
        st.error(f"Smallest class ({n_minority} samples) is too small for CV (need >= 2 splits).")
        return None, None, None
    else:
        # Calculate k based on minimum expected samples in training folds
        max_minority_in_test_fold = np.ceil(n_minority / n_splits)
        min_minority_in_training_data = n_minority - max_minority_in_test_fold

        if min_minority_in_training_data >= 2:
            # Set k to be 1 less than the minimum expected
            smote_k_neighbors = max(1, int(min_minority_in_training_data - 1))
            st.info(f"Using SMOTE with k_neighbors = {smote_k_neighbors} for CV.")
            smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
            pipeline_cv = ImbPipeline(steps=[('preprocessor', preprocessor), ('smote', smote), ('classifier', model)])
            smote_used = True
        else:
            st.warning(f"Not enough minority samples ({int(min_minority_in_training_data)}) expected in training folds for SMOTE k=1. Training without SMOTE.")
            pipeline_cv = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            smote_used = False

    # --- Stratified K-Fold Cross-Validation (Matching Notebook Part 3.4) ---
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision_class1': make_scorer(precision_score, pos_label=1, zero_division=0),
        'recall_class1': make_scorer(recall_score, pos_label=1, zero_division=0),
        'f1_class1': make_scorer(f1_score, pos_label=1, zero_division=0)
    }

    cv_results = None
    cv_successful = False
    avg_metrics = {}
    try:
        st.write(f"Running {n_splits}-Fold Cross-Validation...")
        # Ensure X does not contain unexpected NaNs after feature engineering/selection
        X_imputed_check = SimpleImputer(strategy='median').fit_transform(X.select_dtypes(include=np.number))
        if np.isnan(X_imputed_check).any():
             st.warning("NaNs detected in numerical features even after initial handling. Check feature engineering steps.")
             # Attempt imputation within the function scope before CV for robustness
             num_imputer = SimpleImputer(strategy='median')
             X[numerical_features] = num_imputer.fit_transform(X[numerical_features])
             cat_imputer = SimpleImputer(strategy='most_frequent')
             # Impute categoricals if needed (though OHE handles it later)
             # X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])


        cv_results = cross_validate(pipeline_cv, X, y, cv=skf, scoring=scoring, n_jobs=-1, error_score='raise')
        cv_successful = True
        avg_metrics = {
            "Avg Accuracy": np.mean(cv_results['test_accuracy']),
            "Avg Precision (Class 1)": np.mean(cv_results['test_precision_class1']),
            "Avg Recall (Class 1)": np.mean(cv_results['test_recall_class1']),
            "Avg F1-Score (Class 1)": np.mean(cv_results['test_f1_class1'])
        }
        st.success("Cross-validation complete.")
        # Optionally display fold results
        # st.dataframe(pd.DataFrame(cv_results))

    except ValueError as e:
        st.error(f"Error during cross-validation: {e}")
        st.error("This often happens with SMOTE and very few minority samples per fold.")
    except Exception as e:
        st.error(f"An unexpected error occurred during cross-validation: {e}")

    # --- Feature Importance (Train final model on all data) ---
    feature_importance_df = pd.DataFrame()
    final_pipeline_trained = None
    if cv_successful:
        try:
            st.write("Training final model on all data for feature importances...")
            # Re-create the pipeline used in CV
            if smote_used:
                 smote_final = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
                 final_pipeline_trained = ImbPipeline(steps=[('preprocessor', preprocessor), ('smote', smote_final), ('classifier', model)])
            else:
                 final_pipeline_trained = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

            final_pipeline_trained.fit(X, y)

            classifier = final_pipeline_trained.named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                preprocessor_fitted = final_pipeline_trained.named_steps['preprocessor']
                # Check if 'cat' transformer exists before accessing it
                if 'cat' in preprocessor_fitted.named_transformers_:
                    cat_transformer_fitted = preprocessor_fitted.named_transformers_['cat']
                    ohe_step = cat_transformer_fitted.named_steps['onehot']
                    if categorical_features: ohe_feature_names = list(ohe_step.get_feature_names_out(categorical_features))
                    else: ohe_feature_names = []
                else: # No categorical features were processed
                    ohe_feature_names = []

                # Check if 'num' transformer exists
                if 'num' in preprocessor_fitted.named_transformers_:
                     num_features_actual = numerical_features # Assuming order is preserved
                else:
                     num_features_actual = []


                all_feature_names = list(num_features_actual) + ohe_feature_names
                importances = classifier.feature_importances_

                if len(all_feature_names) == len(importances):
                    feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
                    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
                else: st.warning(f"Feature name/importance mismatch ({len(all_feature_names)} vs {len(importances)}). Cannot show importances.")
        except Exception as e:
            st.warning(f"Could not train final model or get feature importances: {e}")

    return avg_metrics, feature_importance_df, final_pipeline_trained

# --- Forecast Function (Unchanged) ---
@st.cache_data
def run_forecast(demand_hash, _hourly_demand):
    if _hourly_demand is None or _hourly_demand.empty or len(_hourly_demand) < 3:
        st.warning("Insufficient data for time series forecast.")
        return None, None
    try:
        _hourly_demand = _hourly_demand.asfreq('H', fill_value=0)
        model_ses = SimpleExpSmoothing(_hourly_demand, initialization_method="estimated").fit(optimized=True)
        forecast_periods = 6
        forecast = model_ses.forecast(steps=forecast_periods)
        return model_ses, forecast
    except ValueError as ve:
         st.warning(f"Could not fit time series model (ValueError): {ve}. Data might lack variation or have issues.")
         return None, None
    except Exception as e:
        st.warning(f"Could not fit or forecast time series: {e}")
        return None, None

# --- Main App ---
st.set_page_config(layout="wide", page_title="Ride-Hailing Analysis Dashboard")
st.title("Ride-Hailing Data Analysis Dashboard")

# --- Sidebar ---
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your ride data (CSV)", type=["csv"])

# --- Load Data & Features ---
file_hash = get_file_hash(uploaded_file)
# Now load_and_process_data includes the advanced feature engineering
df_active, user_summary, driver_summary, hourly_demand, data_source_msg = load_and_process_data(file_hash, uploaded_file)
st.sidebar.info(f"Displaying analysis for: {data_source_msg}")

if df_active is None:
    st.warning("Please upload a valid CSV file or use the default sample data.")
    st.stop()

if uploaded_file is None:
     st.markdown("*:warning: Displaying analysis based on the **default sample dataset** (20 rides). Insights are illustrative.*")
else:
     st.markdown(f"*:information_source: Displaying analysis based on the **uploaded file:** `{uploaded_file.name}`.*")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview & EDA", "User Behavior", "Cancellation Model", "Operational Insights & Forecast"])

# --- Tab 1: Overview & EDA ---
with tab1:
    st.header("Dataset Overview & EDA")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rides", df_active.shape[0])
    if 'start_time' in df_active.columns and not df_active['start_time'].empty:
         col2.metric("Date Range", f"{df_active['start_time'].min().strftime('%Y-%m-%d')} to {df_active['start_time'].max().strftime('%Y-%m-%d')}")
    col3.metric("Unique Users", df_active['user_id'].nunique())
    st.subheader("Processed Data Sample (with engineered features)")
    st.dataframe(df_active.head(20)) # Show fewer rows now that there are many columns

    st.subheader("Distributions of Key Features")
    # Keep this simple, using original columns before complex engineering if preferred
    dist_cols_options = ['fare', 'distance_km', 'rating', 'trip_duration_minutes']
    dist_col_select = st.selectbox("Select Feature:", [c for c in dist_cols_options if c in df_active.columns], key='dist_select')
    if dist_col_select and df_active[dist_col_select].notna().any():
            fig_dist, ax_dist = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df_active[dist_col_select].dropna(), kde=True, ax=ax_dist[0])
            ax_dist[0].set_title(f'Distribution of {dist_col_select}')
            sns.boxplot(y=df_active[dist_col_select].dropna(), ax=ax_dist[1])
            ax_dist[1].set_title(f'Box Plot of {dist_col_select}')
            plt.tight_layout(); st.pyplot(fig_dist); plt.close(fig_dist)
    elif dist_col_select: st.warning(f"Column '{dist_col_select}' contains only missing values.")
    # ... (rest of EDA plots can remain similar, using basic time/location features) ...
    st.subheader("Ride Volume Over Time")
    fig_vol, (ax_hr, ax_day) = plt.subplots(1, 2, figsize=(14, 5)); plot_vol_possible = False
    if 'start_hour' in df_active.columns and df_active['start_hour'].notna().any():
        sns.countplot(data=df_active, x='start_hour', ax=ax_hr, palette='viridis'); ax_hr.set_title('Rides per Hour'); plot_vol_possible = True
    if 'start_dayname' in df_active.columns and df_active['start_dayname'].notna().any():
         days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
         present_days = df_active['start_dayname'].dropna().unique(); order = [d for d in days_order if d in present_days]
         if order: sns.countplot(data=df_active, x='start_dayname', order=order, ax=ax_day, palette='magma'); ax_day.set_title('Rides per Day'); ax_day.tick_params(axis='x', rotation=45); plot_vol_possible = True
         else: ax_day.set_title('No Day Data')
    if plot_vol_possible: plt.tight_layout(); st.pyplot(fig_vol)
    else: st.warning("Could not plot ride volume.");
    plt.close(fig_vol)

    st.subheader("Top Locations")
    fig_loc, (ax_pick, ax_drop) = plt.subplots(1, 2, figsize=(14, 5)); plot_loc_possible = False; n_top = 10
    if 'pickup_location' in df_active.columns:
        top_pickups = df_active['pickup_location'].value_counts().nlargest(n_top)
        if not top_pickups.empty: sns.barplot(x=top_pickups.values, y=top_pickups.index, ax=ax_pick, palette='Blues_d', orient='h'); ax_pick.set_title(f'Top {min(n_top, len(top_pickups))} Pickups'); plot_loc_possible = True
    if 'dropoff_location' in df_active.columns:
        top_dropoffs = df_active['dropoff_location'].value_counts().nlargest(n_top)
        if not top_dropoffs.empty: sns.barplot(x=top_dropoffs.values, y=top_dropoffs.index, ax=ax_drop, palette='Greens_d', orient='h'); ax_drop.set_title(f'Top {min(n_top, len(top_dropoffs))} Dropoffs'); plot_loc_possible = True
    if plot_loc_possible: plt.tight_layout(); st.pyplot(fig_loc)
    else: st.warning("Could not plot locations.");
    plt.close(fig_loc)

# --- Tab 2: User Behavior ---
with tab2:
    st.header("User Behavior Analysis")
    if user_summary is not None and not user_summary.empty:
        st.subheader("User Summary Statistics")
        format_dict = {}
        if 'avg_rating' in user_summary.columns: format_dict['avg_rating'] = "{:.2f}"
        if 'total_spending' in user_summary.columns: format_dict['total_spending'] = "${:,.2f}"
        if 'recency_hours' in user_summary.columns: format_dict['recency_hours'] = "{:.1f} hrs"
        st.dataframe(user_summary.style.format(format_dict))
        st.subheader("User Activity Distributions")
        fig_user, ax_user = plt.subplots(1, 2, figsize=(14, 5)); plot_user_possible = False
        if 'total_rides' in user_summary.columns:
            bins_rides = min(15, max(5, user_summary['total_rides'].nunique())); sns.histplot(user_summary['total_rides'], bins=bins_rides, kde=False, ax=ax_user[0]); ax_user[0].set_title('Rides per User'); plot_user_possible = True
        else: ax_user[0].set_title('Total Rides N/A')
        if 'recency_hours' in user_summary.columns and user_summary['recency_hours'].notna().any():
            sns.histplot(user_summary['recency_hours'].dropna(), bins=10, kde=True, ax=ax_user[1]); ax_user[1].set_title('User Recency (Hours)'); plot_user_possible = True
        else: ax_user[1].set_title('Recency N/A')
        if plot_user_possible: plt.tight_layout(); st.pyplot(fig_user)
        else: st.warning("Could not plot user distributions.");
        plt.close(fig_user)
    else: st.warning("User summary data could not be generated.")

# --- Tab 3: Cancellation Model (Uses Notebook Logic) ---
with tab3:
    st.header("Cancellation Prediction Model")
    st.markdown("Using **Stratified Cross-Validation** and features from the analysis notebook.")

    df_active_hash = get_df_hash(df_active)
    # This function now contains the notebook's Part 3 logic
    avg_cv_metrics, feat_imp_df, _ = train_evaluate_cv_model(df_active_hash, df_active)

    if avg_cv_metrics:
        st.subheader("Average Cross-Validation Performance")
        col1, col2, col3, col4 = st.columns(4) # Add column for Accuracy
        col1.metric("Avg Accuracy", f"{avg_cv_metrics.get('Avg Accuracy', 0):.2f}")
        col2.metric("Avg Precision (Cancelled=1)", f"{avg_cv_metrics.get('Avg Precision (Class 1)', 0):.2f}")
        col3.metric("Avg Recall (Cancelled=1)", f"{avg_cv_metrics.get('Avg Recall (Class 1)', 0):.2f}")
        col4.metric("Avg F1-Score (Cancelled=1)", f"{avg_cv_metrics.get('Avg F1-Score (Class 1)', 0):.2f}")

        recall_val = avg_cv_metrics.get('Avg Recall (Class 1)', 0)
        if recall_val == 0 and df_active['cancelled'].sum() > 0: st.error("Model failed to predict ANY cancellations during cross-validation.")
        elif df_active['cancelled'].sum() == 0: st.info("No cancellations present in the loaded data.")
        elif recall_val < 0.1: st.warning(f"Model recall for cancellations ({recall_val:.2f}) is very low.")

        st.subheader("Top Feature Importances (from Final Model)")
        if feat_imp_df is not None and not feat_imp_df.empty:
            fig_imp, ax_imp = plt.subplots(figsize=(10, 8)) # Make taller for more features
            n_feat_plot = min(20, len(feat_imp_df)) # Show more features
            sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(n_feat_plot), palette='viridis', ax=ax_imp)
            ax_imp.set_title(f'Top {n_feat_plot} Feature Importances')
            plt.tight_layout(); st.pyplot(fig_imp); plt.close(fig_imp)
        else: st.warning("Could not determine feature importances.")
    else: st.error("Model could not be evaluated using Cross-Validation. Check data/logs.")

# --- Tab 4: Operational Insights & Forecast ---
with tab4:
    st.header("Operational Insights & Forecast (Illustrative)")
    st.subheader("Driver Summary")
    if driver_summary is not None and not driver_summary.empty:
         format_dict_driver = {}
         if 'avg_rating_received' in driver_summary.columns: format_dict_driver['avg_rating_received'] = "{:.2f}"
         if 'cancellation_rate' in driver_summary.columns: format_dict_driver['cancellation_rate'] = "{:.1%}"
         st.dataframe(driver_summary.style.format(format_dict_driver))
    else: st.warning("Driver summary data could not be generated.")

    st.subheader("Demand Forecast (Simple Exponential Smoothing)")
    demand_hash = get_df_hash(hourly_demand.to_frame() if hourly_demand is not None else None)
    ses_model, forecast_vals = run_forecast(demand_hash, hourly_demand)
    if ses_model is not None and forecast_vals is not None:
        fig_fc, ax_fc = plt.subplots(figsize=(12, 6))
        ax_fc.plot(hourly_demand.index, hourly_demand, label='Historical', marker='.')
        ax_fc.plot(forecast_vals.index, forecast_vals, label='Forecast', marker='x', linestyle='--')
        ax_fc.plot(ses_model.fittedvalues.index, ses_model.fittedvalues, label='Fitted', linestyle=':', color='grey')
        ax_fc.set_title('Illustrative Hourly Demand Forecast'); ax_fc.set_ylabel('Rides'); ax_fc.legend(); ax_fc.grid(True)
        plt.tight_layout(); st.pyplot(fig_fc); plt.close(fig_fc)
        st.markdown("*:warning: Forecast uses a simple model. Reliability depends heavily on data.*")
    else: st.warning("Could not generate time series forecast.")

    st.subheader("Key Recommendations (General)")
    st.markdown("""
    *   **Data Quality & Volume:** Ensure sufficient, high-quality data.
    *   **Understand Cancellations:** Collect reasons.
    *   **Context is Key:** Incorporate external factors.
    *   **Monitor & Iterate:** Continuously monitor KPIs.
    *   **A/B Testing:** Validate changes rigorously.
    """)

# --- Footer ---
st.markdown("---")
st.caption("Dashboard created for Ride-Hailing Data Scientist Take-Home Test.")