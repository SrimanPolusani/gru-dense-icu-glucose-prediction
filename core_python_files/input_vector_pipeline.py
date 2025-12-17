import sqlite3
import pandas as pd
import numpy as np

import os
from pasthistory_encoder import PastHistoryGlycemicEncoder
from apachedx_encoder import ApacheDxEncoder

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataExtractor:
    """
    Handles connection to the eICU SQLite database and extraction of raw data tables.
    
    Purpose:
        To serve as the single interface for querying static and dynamic patient data
        from the 'eicu_v2_0_1.sqlite3' database.
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            logging.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logging.error(f"Database connection error: {e}")

    def close(self):
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")

    def get_static_data(self, patient_ids=None):
        """
        Fetches static patient features from the 'patient' and 'apachePatientResult' tables.
        
        Input:
            patient_ids (list, optional): List of patientunitstayid integers to filter by.
                                          If None, fetches all available patients.
                                          
        Source:
            - Table: 'patient' (p)
            - Table: 'apachePatientResult' (a)
            
        Actions:
            1. Connects to the database if not already connected.
            2. Constructs a SQL query to join 'patient' and 'apachePatientResult'.
            3. Selects columns: age, gender, ethnicity, admissionheight, admissionweight, 
               apacheadmissiondx, apachescore.
            4. Filters by patient_ids if provided.
            5. Executes query and returns result as a DataFrame.
            
        Output:
            pd.DataFrame: DataFrame containing raw static features for each patient.
            
        Purpose:
            To retrieve baseline patient characteristics and admission severity scores
            needed for the static input vector.
        """
        if not self.conn:
            self.connect()
        
        query = f"""
            SELECT 
                p.patientunitstayid,
                p.age,
                p.gender,
                p.ethnicity,
                p.admissionheight,
                p.admissionweight,
                p.apacheadmissiondx,
                a.apachescore
            FROM patient p
            LEFT JOIN apachePatientResult a ON p.patientunitstayid = a.patientunitstayid
        """
        
        if patient_ids:
            ids_str = ','.join(map(str, patient_ids))
            query += f" WHERE p.patientunitstayid IN ({ids_str})"
        try:
            df = pd.read_sql_query(query, self.conn)
            logging.info(f"Fetched static data for {len(df)} patients.")
            return df
        except Exception as e:
            logging.error(f"Error fetching static data: {e}")
            return pd.DataFrame()

    def get_pasthistory_data(self, patient_ids=None):
        """
        Fetches past medical history paths from the 'pasthistory' table.
        
        Input:
            patient_ids (list, optional): List of patientunitstayid integers to filter by.
            
        Source:
            - Table: 'pasthistory'
            
        Actions:
            1. Connects to database.
            2. Queries 'patientunitstayid' and 'pasthistorypath'.
            3. Filters by patient_ids if provided.
            
        Output:
            pd.DataFrame: DataFrame with columns ['patientunitstayid', 'pasthistorypath'].
            
        Purpose:
            To provide the raw hierarchical text data needed by `PastHistoryGlycemicEncoder`
            to identify pre-existing comorbidities (e.g., Diabetes, Renal Failure).
        """
        if not self.conn:
            self.connect()
            
        query = "SELECT patientunitstayid, pasthistorypath FROM pasthistory"
        
        if patient_ids:
            ids_str = ','.join(map(str, patient_ids))
            query += f" WHERE patientunitstayid IN ({ids_str})"
            
        try:
            df = pd.read_sql_query(query, self.conn)
            logging.info(f"Fetched pasthistory data for {len(df)} rows.")
            return df
        except Exception as e:
            logging.error(f"Error fetching pasthistory data: {e}")
            return pd.DataFrame()

    def get_dynamic_data(self, patient_ids=None):
        """
        Fetches time-series data (Vitals, Labs, Medications, Infusions) from respective tables.
        
        Input:
            patient_ids (list, optional): List of patientunitstayid integers to filter by.
            
        Source:
            - Table: 'vitalperiodic' (Heart Rate, MAP, Respiration, SaO2)
            - Table: 'lab' (Glucose, pH, Lactate, Electrolytes, etc.)
            - Table: 'medication' (Intermittent drugs)
            - Table: 'infusionDrug' (Continuous infusions)
            
        Actions:
            1. Connects to database.
            2. Constructs 4 separate SQL queries for vitals, labs, meds, and infusions.
            3. Filters each by patient_ids if provided.
            4. Selects relevant columns (ID, offset, names, values).
            5. Executes queries and stores results in a dictionary.
            
        Output:
            dict: A dictionary containing 4 DataFrames:
                  - 'vitals': Periodic vital signs.
                  - 'labs': Laboratory results.
                  - 'meds': Intermittent medication administrations.
                  - 'infusions': Continuous drug infusions.
                  
        Purpose:
            To gather all temporal clinical data points required to construct the 
            dynamic input sequences for the RNN model.
        """
        if not self.conn:
            self.connect()
            
        where_clause = ""
        if patient_ids:
            ids_str = ','.join(map(str, patient_ids))
            where_clause = f"WHERE patientunitstayid IN ({ids_str})"
            
        # 1. Vitals
        query_vitals = f"""
            SELECT patientunitstayid, observationoffset, heartrate, systemicmean, respiration, sao2
            FROM vitalperiodic
            {where_clause}
            ORDER BY patientunitstayid, observationoffset
        """
        
        # 2. Labs (Refined List)
        selected_labs = [
            'bedside glucose', 'glucose', 'anion gap', 'bicarbonate', 'Total CO2', 'pH', 'lactate', 'Base Excess',
            'albumin', 'total protein', 'potassium', 'sodium', 'chloride', 'magnesium', 'phosphate', 
            'calcium', 'ionized calcium', 'creatinine', 'BUN', 'ALT (SGPT)', 'AST (SGOT)', 'total bilirubin',
            'WBC x 1000', 'platelets x 1000', 'Temperature', 'troponin - I', 'paO2', 'paCO2', 'O2 Sat (%)'
        ]
        labs_str = "', '".join(selected_labs)
        
        query_labs = f"""
            SELECT patientunitstayid, labresultoffset as observationoffset, labname, labresult
            FROM lab
            {where_clause} 
            AND labname IN ('{labs_str}')
            
            ORDER BY patientunitstayid, labresultoffset
        """
        
        # 3. Medications (Intermittent)
        query_meds = f"""
            SELECT patientunitstayid, drugstartoffset as observationoffset, drugname, dosage
            FROM medication
            {where_clause}
            ORDER BY patientunitstayid, drugstartoffset
        """
        
        # 4. Infusions (Continuous)
        query_infusions = f"""
            SELECT patientunitstayid, infusionoffset as observationoffset, drugname
            FROM infusionDrug
            {where_clause}
            ORDER BY patientunitstayid, infusionoffset
        """
        
        data = {}
        try:
            data['vitals'] = pd.read_sql_query(query_vitals, self.conn)
            data['labs'] = pd.read_sql_query(query_labs, self.conn)
            data['meds'] = pd.read_sql_query(query_meds, self.conn)
            data['infusions'] = pd.read_sql_query(query_infusions, self.conn)
            logging.info(f"Fetched dynamic data: {len(data['vitals'])} vitals, {len(data['labs'])} labs, {len(data['meds'])} meds, {len(data['infusions'])} infusions.")
        except Exception as e:
            logging.error(f"Error fetching dynamic data: {e}")
        
        return data

class Preprocessor:
    """
    Handles cleaning, transformation, and feature engineering of raw clinical data.
    
    Purpose:
        To transform raw DataFrames from `DataExtractor` into clean, normalized, 
        and structured feature sets ready for tensor generation.
    """
    def __init__(self):
        self.scaler_params = {}

    def process_static_features(self, df):
        """
        Cleans and engineers features for the static input vector.
        
        Input:
            df (pd.DataFrame): Raw static data from `DataExtractor.get_static_data`.
            
        Source:
            - Columns: age, gender, ethnicity, admissionheight, admissionweight, 
              apacheadmissiondx, apachescore.
            
        Actions:
            1. Converts numeric columns to float, coercing errors.
            2. Drops duplicate patient IDs.
            3. Calculates BMI from height and weight (handling units and outliers).
            4. Generates binary masks for missing values (Age, ApacheScore, BMI).
            5. Imputes missing numeric values with the median.
            6. One-hot encodes categorical variables (Gender, Ethnicity, ApacheDx).
            7. Normalizes numeric features (Z-score standardization).
            
        Output:
            pd.DataFrame: Processed static features, indexed by original index (or reset),
                          ready for merging with other static components.
                          
        Purpose:
            To prepare a complete, numerical static feature vector for each patient,
            handling missingness and scaling differences.
        """
        if df.empty:
            return df
            
        df = df.copy()
            
        # 0. Drop Duplicates (Fix for join expansion)
        df = df.drop_duplicates(subset=['patientunitstayid'])
            
        # 1. Convert to numeric
        numeric_cols = ['age', 'admissionweight', 'admissionheight', 'apachescore']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 2. Feature Engineering: BMI (Before Imputation)
        # Height is in cm, convert to meters
        height_m = df['admissionheight'] / 100.0
        # Avoid division by zero
        height_m = height_m.replace(0, np.nan)
        
        df['admissionbmi'] = df['admissionweight'] / (height_m ** 2)
        # Handle BMI outliers/errors - set values outside realistic range (10-115) to NaN
        df['admissionbmi'] = df['admissionbmi'].apply(lambda x: x if (10 <= x <= 115) else np.nan if pd.notna(x) else np.nan)
        
        # 3. Generate Masks (1=Present, 0=Missing)
        # We want masks for Age, ApacheScore, and BMI
        mask_cols = ['age', 'apachescore', 'admissionbmi']
        for col in mask_cols:
            df[f'{col}_mask'] = (~df[col].isna()).astype(int)
            
        # 4. Imputation
        # Fill missing numeric with median
        impute_cols = ['age', 'apachescore', 'admissionbmi']
        for col in impute_cols:
             df[col] = df[col].fillna(df[col].median())
        
        # Drop raw height and weight
        df = df.drop(columns=['admissionheight', 'admissionweight'])

        # 5. One-Hot Encoding
        # Note: apacheadmissiondx is NOT included here because it's processed separately
        # by ApacheDxEncoder to create focused binary features (sepsis, DKA, etc.)
        
        # DEFINE FIXED SCHEMA FOR STATIC CATEGORICALS
        EXPECTED_GENDERS = ['Female', 'Male'] # 'Other' often dropped or mapped, but let's be explicit if needed. 
        # Actually, get_dummies usually drops one if drop_first=True, but here we kept all.
        # Let's assume we want specific columns.
        
        # To match typical eICU data:
        # Gender: Female, Male
        # Ethnicity: Caucasian, African American, Asian, Hispanic, Native American, Other/Unknown
        
        # We will perform get_dummies then reindex.
        categorical_cols = ['gender', 'ethnicity']
        df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)
        
        # ENFORCE STATIC SCHEMA
        # We need to explicitly list the columns we EXPECT to see after get_dummies
        # This ensures that if "Native American" is missing from a batch, the column is created with 0s.
        
        expected_static_dummies = [
            'gender_Female', 'gender_Male', 
            'ethnicity_African American', 'ethnicity_Asian', 'ethnicity_Caucasian', 
            'ethnicity_Hispanic', 'ethnicity_Native American', 'ethnicity_Other/Unknown'
        ]
        
        # Add dummy_na columns if we want them (we used dummy_na=True above)
        expected_static_dummies += ['gender_nan', 'ethnicity_nan']
        
        # Reindex to enforce presence of all dummy columns
        # We only reindex the categorical parts. We need to keep the numeric parts.
        # So we construct the full expected column list for the dataframe? 
        # Easier: just ensure these specific columns exist.
        
        for col in expected_static_dummies:
            if col not in df.columns:
                df[col] = 0
                
        # Optional: Drop unexpected columns if any (e.g. weird typos in raw data)
        # For now, just ensuring presence is enough to prevent "missing column" errors.
        # But to be safe for concatenation, we should probably sort or reindex the whole thing eventually.
        # For now, let's just ensure the critical ones exist.
        
        # FIX: Convert booleans to int (0/1) to avoid object dtype warnings
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)
        
        # FIX: Drop raw apacheadmissiondx as it is text and handled elsewhere
        if 'apacheadmissiondx' in df.columns:
            df = df.drop(columns=['apacheadmissiondx'])
        
        # 6. Normalization (Z-score)
        # Update numeric cols to include BMI and exclude dropped cols
        # We generally do NOT normalize masks (they are binary 0/1)
        final_numeric_cols = ['age', 'admissionbmi', 'apachescore']
        
        for col in final_numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std
            self.scaler_params[col] = {'mean': mean, 'std': std}
            
        # --- ENFORCE FINAL SCHEMA AND ORDER ---
        # Explicitly define the final column list and order
        # 1. ID
        # 2. Numeric Features
        # 3. Masks
        # 4. Categorical Dummies (Fixed Order)
        
        final_columns = ['patientunitstayid'] + final_numeric_cols + mask_cols + expected_static_dummies
        
        # Reindex forces the columns to match this list exactly.
        # - Missing columns are created with NaN (we fill with 0)
        # - Extra columns are dropped
        # - Order is enforced
        df = df.reindex(columns=final_columns, fill_value=0)
            
        logging.info(f"Processed static features. Shape: {df.shape}")
        return df

    def process_pasthistory(self, df):
        """
        Encodes past history text paths into patient-level binary flags.
        
        Input:
            df (pd.DataFrame): Raw past history data from `DataExtractor.get_pasthistory_data`.
            
        Source:
            - Column: 'pasthistorypath'
            - Helper: `PastHistoryGlycemicEncoder` class.
            
        Actions:
            1. Instantiates `PastHistoryGlycemicEncoder`.
            2. Applies the encoder to the DataFrame to get binary columns.
            3. Groups by 'patientunitstayid' and takes the MAX value.
               (If a patient has *any* record indicating a condition, they are flagged).
            
        Output:
            pd.DataFrame: Aggregated DataFrame with one row per patient and 11 binary columns.
            
        Purpose:
            To condense multiple history records per patient into a single fixed-size 
            vector of comorbidities.
        """
        if df.empty:
            return pd.DataFrame()
            
        encoder = PastHistoryGlycemicEncoder()
        features = encoder.transform(df)
        
        # Combine with ID
        features['patientunitstayid'] = df['patientunitstayid']
        
        # Group by ID and take Max (if any path has the condition, the patient has it)
        aggregated = features.groupby('patientunitstayid').max().reset_index()
        
        logging.info(f"Processed pasthistory features. Shape: {aggregated.shape}")
        return aggregated

    def process_apachedx(self, df):
        """
        Encodes admission diagnoses into patient-level binary flags.
        
        Input:
            df (pd.DataFrame): Raw static data containing 'apacheadmissiondx' 
                               (usually passed from `process_static_features` or raw source).
            
        Source:
            - Column: 'apacheadmissiondx'
            - Helper: `ApacheDxEncoder` class.
            
        Actions:
            1. Instantiates `ApacheDxEncoder`.
            2. Applies the encoder to get binary columns for high-impact diagnoses.
            3. Ensures uniqueness by dropping duplicate patient IDs.
            
        Output:
            pd.DataFrame: DataFrame with 'patientunitstayid' and 8 binary diagnosis columns.
            
        Purpose:
            To extract specific, high-priority admission reasons (e.g., Sepsis, DKA) 
            that are critical for glycemic prediction but might be lost in general one-hot encoding.
        """
        if df.empty:
            return pd.DataFrame()
            
        encoder = ApacheDxEncoder()
        features = encoder.transform(df)
        
        # Combine with ID
        features['patientunitstayid'] = df['patientunitstayid']
        
        # FIX: Ensure uniqueness (handle duplicates from static_df)
        features = features.groupby('patientunitstayid').max().reset_index()
        
        logging.info(f"Processed apachedx features. Shape: {features.shape}")
        return features

    def _map_drug_category(self, drug_name):
        """
        Helper function to map raw drug names to high-level pharmacological categories.
        
        Input:
            drug_name (str): The name of the medication or infusion.
            
        Source:
            - `medication` table ('drugname')
            - `infusionDrug` table ('drugname')
            
        Actions:
            1. Normalizes string to lowercase.
            2. Checks against predefined keyword lists for 11 categories (e.g., Insulin, Corticosteroids).
            3. Returns the matching category name or 'Other'.
            
        Output:
            str: The mapped category name.
            
        Purpose:
            To reduce the high dimensionality of medication names into clinically meaningful 
            groups relevant to glucose metabolism.
        """
        if pd.isna(drug_name):
            return 'Other'
            
        name = str(drug_name).lower()
        
        # High Impact Categories for Glycemic Control
        
        # 1. Insulin (Drips & Shots)
        if any(x in name for x in ['insulin', 'humalog', 'novolog', 'lantus', 'lispro', 'glargine', 'regular', 'detemir', 'nph', 'aspart', 'levemir', 'humulin']):
            return 'Insulin'
            
        # 2. Dextrose - Rescue (Hypo Treatment) vs Maintenance
        if any(x in name for x in ['glucagon', 'glucagen']):
            return 'Glucagon' # Critical signal for instability
        if any(x in name for x in ['dextrose 50%', 'd50', 'glutose']):
            return 'Dextrose_Rescue'
        if any(x in name for x in ['dextrose', 'd5w', 'd10', 'd5', 'glucose', 'dext']):
            return 'Dextrose_Maintenance'
            
        # 3. Vasopressors (Stress Response)
        if any(x in name for x in ['norepinephrine', 'epinephrine', 'vasopressin', 'phenylephrine', 'dopamine', 'levophed', 'adrenaline', 'neo-synephrine', 'ephedrine']):
            return 'Vasopressor'
            
        # 4. Corticosteroids (Hyperglycemia Inducing)
        if any(x in name for x in ['hydrocortisone', 'prednisone', 'methylprednisolone', 'dexamethasone', 'solu-medrol', 'solu-cortef', 'decadron', 'medrol', 'solumedrol', 'deltasone']):
            return 'Corticosteroid'
            
        # 5. Beta Blockers (Masks Hypo)
        if any(x in name for x in ['metoprolol', 'labetalol', 'carvedilol', 'atenolol', 'esmolol', 'lopressor', 'trandate', 'coreg']):
            return 'BetaBlocker'
            
        # 6. Sedatives (Propofol has calories)
        if any(x in name for x in ['propofol', 'diprivan', 'midazolam', 'versed', 'fentanyl', 'sublimaze', 'dexmedetomidine', 'precedex']):
            return 'Sedative_Propofol'
            
        # 7. Nutrition (High Glucose Load)
        if any(x in name for x in ['tpn', 'total parenteral nutrition', 'clinimix', 'kabiven', 'ensure', 'jevity', 'nepro', 'fat emulsion', 'lipids']):
            return 'Nutrition'
            
        # 8. Oral Hypoglycemics (Diabetes History)
        if any(x in name for x in ['metformin', 'glipizide', 'glyburide', 'glimepiride', 'sitagliptin', 'januvia', 'actos', 'avandia']):
            return 'OralHypoglycemic'
            
        # 9. Hypo-Inducing Antibiotics
        if any(x in name for x in ['ciprofloxacin', 'levofloxacin', 'moxifloxacin', 'gatifloxacin', 'bactrim', 'sulfamethoxazole', 'pentamidine', 'levaquin']):
            return 'HypoAntibiotic'
            
        # 10. Sympathomimetics
        if any(x in name for x in ['albuterol', 'salbutamol', 'ventolin', 'proventil']):
            return 'Sympathomimetic'
            
        # 11. Octreotide
        if any(x in name for x in ['octreotide', 'sandostatin']):
            return 'Octreotide'
            
        return 'Other'

    def process_dynamic_features(self, dynamic_data):
        """
        Processes and aligns all time-series data into a unified hourly format.
        
        Input:
            dynamic_data (dict): Dictionary of DataFrames (vitals, labs, meds, infusions) 
                                 from `DataExtractor.get_dynamic_data`.
                                 
        Source:
            - 'vitals', 'labs', 'meds', 'infusions' DataFrames.
            
        Actions:
            1. Iterates through each unique patient.
            2. Vitals: Resamples to 1-hour means.
            3. Labs: Pivots to wide format, merges Glucose/Bicarbonate variants, resamples to 1-hour means.
            4. Meds/Infusions: Concatenates, maps to categories, pivots to wide format (presence/absence), resamples to 1-hour max.
            5. Merges all streams on the time index (outer join).
            6. Generates binary masks for missing data.
            7. Imputes missing values (Forward Fill -> Backward Fill -> Fill 0).
            8. Concatenates features and masks.
            
        Output:
            pd.DataFrame: A single DataFrame containing processed hourly data for all patients, 
                          stacked vertically.
                          
        Purpose:
            To create a clean, regularly sampled time-series dataset for every patient, 
            handling the irregularity and sparsity of raw clinical data.
        """
        vitals = dynamic_data.get('vitals', pd.DataFrame())
        labs = dynamic_data.get('labs', pd.DataFrame())
        meds = dynamic_data.get('meds', pd.DataFrame())
        infusions = dynamic_data.get('infusions', pd.DataFrame())
        
        processed_patients = []
        
        # Get all unique patient IDs from all sources
        all_pids = set()
        if not vitals.empty: all_pids.update(vitals['patientunitstayid'].unique())
        if not labs.empty: all_pids.update(labs['patientunitstayid'].unique())
        if not meds.empty: all_pids.update(meds['patientunitstayid'].unique())
        if not infusions.empty: all_pids.update(infusions['patientunitstayid'].unique())
        
        for pid in all_pids:
            # --- VITALS ---
            if not vitals.empty:
                p_vitals = vitals[vitals['patientunitstayid'] == pid].copy()
            else:
                p_vitals = pd.DataFrame()

            if not p_vitals.empty:
                p_vitals['observationoffset'] = pd.to_numeric(p_vitals['observationoffset'], errors='coerce')
                p_vitals = p_vitals.set_index('observationoffset')
                p_vitals.index = pd.to_timedelta(p_vitals.index, unit='m')
                p_vitals = p_vitals.drop(columns=['patientunitstayid'])
                # Convert to numeric to avoid TypeError during resampling
                p_vitals = p_vitals.apply(pd.to_numeric, errors='coerce')
                # Resample to 1 hour, taking mean
                p_vitals = p_vitals.resample('1h').mean()
            else:
                p_vitals = pd.DataFrame()

            # --- LABS ---
            if not labs.empty:
                p_labs = labs[labs['patientunitstayid'] == pid].copy()
            else:
                p_labs = pd.DataFrame()

            if not p_labs.empty:
                p_labs['observationoffset'] = pd.to_numeric(p_labs['observationoffset'], errors='coerce')
                p_labs['labresult'] = pd.to_numeric(p_labs['labresult'], errors='coerce')
                
                # Pivot first to get columns for each lab
                p_labs_pivot = p_labs.pivot_table(index='observationoffset', columns='labname', values='labresult', aggfunc='mean')
                
                # --- MERGING LOGIC ---
                # 1. Glucose Merge
                if 'glucose' in p_labs_pivot.columns and 'bedside glucose' in p_labs_pivot.columns:
                    p_labs_pivot['glucose_level'] = p_labs_pivot['glucose'].fillna(p_labs_pivot['bedside glucose'])
                elif 'glucose' in p_labs_pivot.columns:
                    p_labs_pivot['glucose_level'] = p_labs_pivot['glucose']
                elif 'bedside glucose' in p_labs_pivot.columns:
                    p_labs_pivot['glucose_level'] = p_labs_pivot['bedside glucose']
                else:
                    p_labs_pivot['glucose_level'] = np.nan
                    
                # 2. Bicarbonate Merge
                if 'bicarbonate' in p_labs_pivot.columns and 'Total CO2' in p_labs_pivot.columns:
                    p_labs_pivot['bicarbonate_merged'] = p_labs_pivot['bicarbonate'].fillna(p_labs_pivot['Total CO2'])
                elif 'bicarbonate' in p_labs_pivot.columns:
                    p_labs_pivot['bicarbonate_merged'] = p_labs_pivot['bicarbonate']
                elif 'Total CO2' in p_labs_pivot.columns:
                    p_labs_pivot['bicarbonate_merged'] = p_labs_pivot['Total CO2']
                else:
                    p_labs_pivot['bicarbonate_merged'] = np.nan
                
                p_labs_pivot.index = pd.to_timedelta(p_labs_pivot.index, unit='m')
                p_labs = p_labs_pivot.resample('1h').mean()
            else:
                p_labs = pd.DataFrame()

            # --- MEDS & INFUSIONS ---
            # Combine both sources
            if not meds.empty:
                p_meds = meds[meds['patientunitstayid'] == pid].copy()
            else:
                p_meds = pd.DataFrame()
                
            if not infusions.empty:
                p_infusions = infusions[infusions['patientunitstayid'] == pid].copy()
            else:
                p_infusions = pd.DataFrame()
             
            # Standardize columns for concatenation
            if not p_meds.empty:
                p_meds = p_meds[['observationoffset', 'drugname']]
            else:
                p_meds = pd.DataFrame(columns=['observationoffset', 'drugname'])
                
            if not p_infusions.empty:
                p_infusions = p_infusions[['observationoffset', 'drugname']]
            else:
                p_infusions = pd.DataFrame(columns=['observationoffset', 'drugname'])
                
            # Concatenate
            p_all_drugs = pd.concat([p_meds, p_infusions])
            
            if not p_all_drugs.empty:
                p_all_drugs['observationoffset'] = pd.to_numeric(p_all_drugs['observationoffset'], errors='coerce')
                
                # Apply robust keyword mapping
                p_all_drugs['category'] = p_all_drugs['drugname'].apply(self._map_drug_category)
                
                # Filter out 'Other' to keep features focused and reduce noise
                p_all_drugs = p_all_drugs[p_all_drugs['category'] != 'Other']
                
                if not p_all_drugs.empty:
                    # Pivot on CATEGORY
                    p_all_drugs['presence'] = 1
                    # Use max to capture presence in the hour
                    p_all_drugs = p_all_drugs.pivot_table(index='observationoffset', columns='category', values='presence', aggfunc='max')
                    p_all_drugs.index = pd.to_timedelta(p_all_drugs.index, unit='m')
                    p_all_drugs = p_all_drugs.resample('1h').max().fillna(0)
                else:
                    p_all_drugs = pd.DataFrame()
            else:
                p_all_drugs = pd.DataFrame()

            # --- MERGE ---
            # Join all on index (time)
            
            # DEFINE FIXED SCHEMA TO ENSURE CONSISTENT DIMENSIONS
            # 1. Vitals
            EXPECTED_VITALS = ['heartrate', 'systemicmean', 'respiration', 'sao2']
            
            # 2. Labs (Raw + Merged)
            # Must match the selected_labs list in get_dynamic_data + the 2 merged columns
            EXPECTED_LABS = [
                'bedside glucose', 'glucose', 'anion gap', 'bicarbonate', 'Total CO2', 'pH', 'lactate', 'Base Excess',
                'albumin', 'total protein', 'potassium', 'sodium', 'chloride', 'magnesium', 'phosphate', 
                'calcium', 'ionized calcium', 'creatinine', 'BUN', 'ALT (SGPT)', 'AST (SGOT)', 'total bilirubin',
                'WBC x 1000', 'platelets x 1000', 'Temperature', 'troponin - I', 'paO2', 'paCO2', 'O2 Sat (%)',
                'glucose_level', 'bicarbonate_merged'
            ]
            
            # 3. Meds (Categories)
            EXPECTED_MEDS = [
                'Insulin', 'Glucagon', 'Dextrose_Rescue', 'Dextrose_Maintenance', 'Vasopressor', 
                'Corticosteroid', 'BetaBlocker', 'Sedative_Propofol', 'Nutrition', 'OralHypoglycemic', 
                'HypoAntibiotic', 'Sympathomimetic', 'Octreotide'
            ]
            
            ALL_EXPECTED_FEATURES = EXPECTED_VITALS + EXPECTED_LABS + EXPECTED_MEDS
            
            p_merged = p_vitals.join([p_labs, p_all_drugs], how='outer')
            
            # --- ENFORCE FIXED SCHEMA ---
            # Reindex to ensure ALL columns exist and are in the correct order
            p_merged = p_merged.reindex(columns=ALL_EXPECTED_FEATURES)
            
            # Fill missing Meds with 0 (since they are binary presence/absence)
            # Vitals and Labs remain NaN to be handled by imputation/masking
            p_merged[EXPECTED_MEDS] = p_merged[EXPECTED_MEDS].fillna(0)
            
            # --- GENERATE MASKS (Before Imputation) ---
            # Create mask: 1 = Present/Observed, 0 = Missing/Imputed
            p_mask = (~p_merged.isna()).astype(int)
            # Rename mask columns with _mask suffix
            p_mask.columns = [f'{col}_mask' for col in p_mask.columns]
            
            # --- IMPUTATION ---
            # FIX: Prevent Data Leakage by removing global bfill/ffill for Vitals/Labs.
            # Only fill Medications/Infusions with 0 (as NaN implies not given).
            # Note: Meds are already filled above, but keeping logic consistent.
            
            # --- CONCATENATE FEATURES AND MASKS ---
            p_merged = pd.concat([p_merged, p_mask], axis=1)
            
            # Add patient ID back
            p_merged['patientunitstayid'] = pid
            
            processed_patients.append(p_merged)
            
        if not processed_patients:
            return pd.DataFrame()
            
        full_df = pd.concat(processed_patients)
        logging.info(f"Processed dynamic features. Shape: {full_df.shape}")
        return full_df

class TensorGenerator:
    """
    Converts processed DataFrames into NumPy arrays (tensors) for Deep Learning models.
    
    Purpose:
        To format the data into (Samples, TimeSteps, Features) for RNNs and 
        (Samples, Features) for Dense networks, aligned by patient and time window.
    """
    def __init__(self, sequence_length=24, prediction_gap=5, prediction_window=2):
        self.sequence_length = sequence_length  # Hours 0-24 (input)
        self.prediction_gap = prediction_gap    # 5 hours gap after input
        self.prediction_window = prediction_window  # Hours 29-30 (target: 5-6 hours ahead)

    def create_sequences(self, dynamic_df):
        """
        Splits continuous patient time-series into fixed-length sequences.
        
        Input:
            dynamic_df (pd.DataFrame): Processed dynamic data from `Preprocessor.process_dynamic_features`.
            
        Source:
            - All dynamic feature columns.
            
        Actions:
            1. Groups data by patient.
            2. Iterates through patient data in sliding windows of length `sequence_length`.
            3. Extracts each window as a sequence.
            
        Output:
            tuple: (sequences, patient_ids)
                   - sequences: 3D np.array (Num_Sequences, Sequence_Length, Num_Features)
                   - patient_ids: 1D np.array of patient IDs corresponding to each sequence.
                   
        Purpose:
            To prepare the sequential input (X_seq) for the RNN/GRU component of the model.
        """
        sequences = []
        patient_ids = []
        
        # Group by patient
        for pid, group in dynamic_df.groupby('patientunitstayid'):
            data = group.drop(columns=['patientunitstayid']).values
            
            if len(data) < self.sequence_length:
                continue
                
            for i in range(len(data) - self.sequence_length):
                seq = data[i : i + self.sequence_length]
                sequences.append(seq)
                patient_ids.append(pid)
                
        return np.array(sequences), np.array(patient_ids)

    def create_static_features(self, static_df, patient_ids_aligned):
        """
        Aligns static features with the generated dynamic sequences.
        
        Input:
            static_df (pd.DataFrame): Processed static data.
            patient_ids_aligned (np.array): List of patient IDs for each sequence 
                                            (from `create_sequences` or `create_dataset`).
                                            
        Source:
            - Static feature vector.
            
        Actions:
            1. Creates a lookup dictionary for static features by patient ID.
            2. Iterates through `patient_ids_aligned`.
            3. Retrieves the corresponding static vector for each sequence.
            
        Output:
            np.array: 2D array (Num_Sequences, Num_Static_Features).
            
        Purpose:
            To provide the static context (X_static) for each dynamic sequence, 
            replicating the static data for every time window of the same patient.
        """
        static_features = []
        
        # Create a lookup dictionary for faster access
        static_dict = static_df.set_index('patientunitstayid').to_dict('index')
        
        for pid in patient_ids_aligned:
            if pid in static_dict:
                # Convert dict values to list
                feats = list(static_dict[pid].values())
                static_features.append(feats)
            else:
                # Should not happen if data is consistent
                static_features.append([0] * (static_df.shape[1] - 1))
                
        return np.array(static_features)

    def create_dataset(self, dynamic_df, static_df, target_col='glucose'):
        """
        Orchestrates the full dataset creation, including inputs and targets.
        
        Input:
            dynamic_df (pd.DataFrame): Processed dynamic data.
            static_df (pd.DataFrame): Processed static data.
            target_col (str): Base name of the target variable (default 'glucose').
            
        Source:
            - Dynamic features (X_seq)
            - Static features (X_static)
            - Future values of 'glucose'/'bedside glucose' (y)
            
        Actions:
            1. Identifies target columns (glucose, bedside glucose).
            2. Iterates through each patient's data.
            3. Slides a window of `sequence_length` (Input) + `prediction_window` (Target).
            4. Extracts Input Sequence (X_seq).
            5. Extracts Static Features for that patient (X_static).
            6. Checks the Target Window for Hyperglycemia (>180) or Hypoglycemia (<70).
            7. Appends to lists and converts to arrays.
            
        Output:
            tuple: (X_seq, X_static, y_hyper, y_hypo)
                   - X_seq: 3D Input Tensor
                   - X_static: 2D Input Tensor
                   - y_hyper: 1D Binary Target (1 if Hyper event in window)
                   - y_hypo: 1D Binary Target (1 if Hypo event in window)
                   
        Purpose:
            To generate the final training/validation datasets for the hybrid model, 
            defining the supervised learning task (predicting future events from past data).
        """
        X_seq = []
        X_static = []
        y_hyper = []
        y_hypo = []
        patient_ids = []
        
        # Static lookup
        static_dict = static_df.set_index('patientunitstayid').to_dict('index')
        
        # Identify target columns (Serum Glucose and Bedside Glucose)
        # We prioritize these specific eICU lab names based on research/standard practice
        possible_targets = ['glucose', 'bedside glucose']
        target_cols = [c for c in dynamic_df.columns if c.lower() in possible_targets]
        
        # Also identify the corresponding mask columns
        target_mask_cols = [f'{c}_mask' for c in target_cols if f'{c}_mask' in dynamic_df.columns]
        
        if not target_cols:
            logging.warning("No glucose columns found ('glucose' or 'bedside glucose'). Labels will be 0.")
        
        for pid, group in dynamic_df.groupby('patientunitstayid'):
            # Drop ID for processing
            data_values = group.drop(columns=['patientunitstayid']).values
            
            # Check if patient has enough data to form at least one sample
            # We need:
            # 1. Enough data for input sequence (sequence_length)
            # 2. At least one data point in the target window (after the gap)
            # We do NOT require data in the gap itself.
            
            # Since we reindexed to hourly frequency, len(data_values) represents the total HOURS of stay.
            # We need the stay to be long enough to reach the target window.
            # Minimum hours = Input Window (24) + Gap (5) + 1 hour of target = 30 hours
            min_span_hours = self.sequence_length + self.prediction_gap + 1
            
            if len(data_values) < min_span_hours:
                continue
                
            # Get static features for this patient
            if pid not in static_dict:
                continue
            s_feats = list(static_dict[pid].values())
            
            # Pre-calculate target indices to avoid doing it inside the inner loop
            cols_after_drop = group.columns.drop('patientunitstayid')
            
            if target_cols:
                target_indices = [cols_after_drop.get_loc(c) for c in target_cols]
                # Get mask indices to check if glucose values are real or imputed
                mask_indices = [cols_after_drop.get_loc(c) for c in target_mask_cols if c in cols_after_drop]
            else:
                target_indices = []
                mask_indices = []

            # Sliding window: create multiple samples from this patient's timeline
            max_start_idx = len(data_values) - self.sequence_length - self.prediction_gap - self.prediction_window
            for i in range(max_start_idx + 1):
                # Input Sequence: Hours 0-24
                # Extract raw data (potentially with NaNs)
                seq_raw = data_values[i : i + self.sequence_length]
                
                # LAZY IMPUTATION: Apply ffill/bfill ONLY to this input window
                # This ensures we don't impute the "gap" data that we don't use
                seq_df = pd.DataFrame(seq_raw)
                seq_filled = seq_df.ffill().bfill().fillna(0).values
                
                seq = seq_filled
                
                # Target Window: Hours 29-30 (5-6 hours from "now" at hour 24)
                if target_indices:
                    # Extract target window: skip 5 hours gap, then get 2-hour window
                    # shape: (prediction_window=2, num_targets)
                    target_start = i + self.sequence_length + self.prediction_gap
                    target_end = target_start + self.prediction_window
                    future_window = data_values[target_start:target_end, target_indices]
                    
                    # CRITICAL: Check if we have REAL glucose measurements (not forward-filled)
                    # Extract the mask values for the same window
                    if mask_indices:
                        future_mask = data_values[target_start:target_end, mask_indices]
                        # If ALL mask values are 0 (all imputed), skip this sample
                        # Use nansum because masks might be NaN if the column was missing for this patient
                        if np.nansum(future_mask) == 0:
                            continue  # No real glucose measurements in target window
                    
                    # Check for events across ALL glucose columns
                    # If ANY value in the window > 180, it's hyper
                    is_hyper = 1 if np.nanmax(future_window) > 180 else 0
                    # If ANY value in the window < 70, it's hypo
                    is_hypo = 1 if np.nanmin(future_window) < 70 else 0
                else:
                    is_hyper = 0
                    is_hypo = 0
                
                X_seq.append(seq)
                X_static.append(s_feats)
                y_hyper.append(is_hyper)
                y_hypo.append(is_hypo)
                # Track Patient ID for each sequence
                patient_ids.append(pid)
                
        return np.array(X_seq), np.array(X_static), np.array(y_hyper), np.array(y_hypo), np.array(patient_ids)
        
def process_chunk(chunk_args):
    """
    Worker function to process a chunk of patient IDs.
    """
    chunk_ids, chunk_idx, db_path, output_dir = chunk_args
    
    # Re-instantiate classes inside the worker process
    # (SQLite connections cannot be pickled/shared across processes)
    extractor = DataExtractor(db_path)
    preprocessor = Preprocessor()
    tensor_gen = TensorGenerator(sequence_length=24, prediction_gap=5, prediction_window=2)
    
    try:
        # --- Static Data ---
        static_df = extractor.get_static_data(chunk_ids)
        if static_df.empty:
            return f"Chunk {chunk_idx}: No static data found."
            
        processed_static = preprocessor.process_static_features(static_df)
        
        # --- Past History ---
        pasthistory_df = extractor.get_pasthistory_data(chunk_ids)
        processed_history = preprocessor.process_pasthistory(pasthistory_df)
        
        # --- Apache Dx ---
        processed_apachedx = preprocessor.process_apachedx(static_df)
        
        # Merge Static
        if not processed_history.empty:
            processed_static = processed_static.merge(processed_history, on='patientunitstayid', how='left')
        if not processed_apachedx.empty:
            processed_static = processed_static.merge(processed_apachedx, on='patientunitstayid', how='left')
            
        processed_static = processed_static.fillna(0)
        
        # Ensure numeric types
        non_numeric = processed_static.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric:
            processed_static[col] = pd.to_numeric(processed_static[col], errors='coerce').fillna(0)
            
        # --- Dynamic Data ---
        dynamic_data = extractor.get_dynamic_data(chunk_ids)
        processed_dynamic = preprocessor.process_dynamic_features(dynamic_data)
        
        if processed_dynamic.empty:
            return f"Chunk {chunk_idx}: No dynamic data found."
            
        # --- Tensor Generation ---
        X_seq, X_static, y_hyper, y_hypo = tensor_gen.create_dataset(processed_dynamic, processed_static)
        
        # Save to disk
        # Save to disk
        if len(X_seq) > 0:
            output_file = os.path.join(output_dir, f"batch_{chunk_idx}.npz")
            try:
                np.savez_compressed(output_file, X_seq=X_seq, X_static=X_static, y_hyper=y_hyper, y_hypo=y_hypo)
                return f"Chunk {chunk_idx}: Saved {len(X_seq)} samples to {output_file}"
            except Exception as e:
                return f"Chunk {chunk_idx}: Error saving file - {str(e)}"
        else:
            return f"Chunk {chunk_idx}: Skipped - No valid samples generated (insufficient data/targets)."
            
    except Exception as e:
        return f"Chunk {chunk_idx}: Error - {str(e)}"
    finally:
        extractor.close()

if __name__ == "__main__":
    import multiprocessing as mp
    import os
    import time
    
    # Configuration
    DB_PATH = 'eicu_v2_0_1.sqlite3'
    OUTPUT_DIR = 'processed_tensors'
    
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"--- Starting Parallel Pipeline ---")
    print(f"Database: {DB_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    
    # 1. Fetch All Patient IDs
    print("Fetching all patient IDs...")
    conn = sqlite3.connect(DB_PATH)
    all_ids = pd.read_sql_query("SELECT patientunitstayid FROM patient", conn)['patientunitstayid'].tolist()
    conn.close()
    
    total_patients = len(all_ids)
    print(f"Total Patients: {total_patients}")
    
    # 2. Configure Multiprocessing
    # Use all available cores
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} Cores")
    
    # Determine Chunk Size
    # Optimization for 64 Cores / 128GB RAM:
    # 128GB / 64 cores = 2GB per core.
    # Conservative estimate: 150 patients per chunk to ensure we stay well under 2GB limit 
    # even with Pandas overhead and potential data density spikes.
    chunk_size = 150
    chunks = [all_ids[i:i + chunk_size] for i in range(0, len(all_ids), chunk_size)]
    print(f"Split into {len(chunks)} chunks of size ~{chunk_size}")
    
    # Prepare arguments for workers
    chunk_args = [(chunk, i, DB_PATH, OUTPUT_DIR) for i, chunk in enumerate(chunks)]
    
    # 3. Run Parallel Processing
    start_time = time.time()
    
    # Use Pool
    with mp.Pool(processes=num_cores) as pool:
        # Use imap_unordered to get results as they finish
        results = pool.imap_unordered(process_chunk, chunk_args)
        
        # Monitor progress
        total_chunks = len(chunks)
        for i, res in enumerate(results):
            if i % 10 == 0:
                print(f"Progress: {i}/{total_chunks} chunks completed. ({res})")
                
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- Processing Complete ---")
    print(f"Time taken: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Results saved in {OUTPUT_DIR}")
