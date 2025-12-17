import re
import pandas as pd

class PastHistoryGlycemicEncoder:
    """
    Encodes patient past history data into binary features relevant to glycemic control.
    
    This class uses regex pattern matching to identify specific clinical conditions
    from the hierarchical text strings found in the 'pasthistorypath' column of the 
    eICU 'pasthistory' table.
    
    Attributes:
        patterns (dict): A dictionary mapping feature names (e.g., 'hx_t1dm') to 
                         regex patterns used for identification.
    """
    def __init__(self):
        # Regex patterns for Glycemic 12
        # Note: Order doesn't strictly matter as we check all, but regex specificity does.
        self.patterns = {
            # T1DM: Matches "Type 1", "IDDM", or "Insulin Dependent" BUT NOT "Non-Insulin Dependent"
            'hx_t1dm': r'(?i)(type\s*1|type\s*I\b|IDDM|(?<!non-)(?<!non\s)insulin\s+dependent)',
            
            # T2DM: Matches "Type 2", "NIDDM", or "Non-Insulin Dependent"
            'hx_t2dm': r'(?i)(type\s*2|type\s*II\b|NIDDM|non-insulin\s+dependent)',
            
            'hx_renal_failure': r'(?i)(renal\s+failure|renal\s+insufficiency|kidney\s+disease)',
            'hx_dialysis': r'(?i)(dialysis|hemodialysis|CAPD)',
            'hx_liver_cirrhosis': r'(?i)(cirrhosis|liver|hepatic)',
            'hx_chf': r'(?i)(congestive\s+heart\s+failure|CHF)',
            'hx_copd': r'(?i)(COPD|emphysema|asthma)',
            'hx_immunosuppression': r'(?i)(immunosuppression|transplant|steroid|chemotherapy)',
            'hx_hypertension': r'(?i)(hypertension)',
            'hx_malignancy': r'(?i)(cancer|malignancy|metastasis|metastases)',
            'hx_arrhythmia': r'(?i)(arrhythmia|fibrillation|tachycardia|pacer|pacemaker)'
        }

    def transform(self, df):
        """
        Transforms the 'pasthistorypath' column into 11 binary features.
        
        Input:
            df (pd.DataFrame): A DataFrame containing the 'pasthistorypath' column. 
                               This data typically comes from the 'pasthistory' table 
                               via `DataExtractor.get_pasthistory_data`.
                               
        Actions:
            1. Validates the input DataFrame contains 'pasthistorypath'.
            2. Converts the column to string type and handles missing values.
            3. Iterates through each defined feature pattern.
            4. Applies regex search to each row's path string.
            5. Creates a binary column (1 if match found, 0 otherwise) for each feature.
            
        Output:
            pd.DataFrame: A new DataFrame with the same index as input, containing 
                          11 binary columns (e.g., 'hx_t1dm', 'hx_chf').
                          
        Purpose:
            To convert unstructured hierarchical text describing past medical history 
            into structured numerical features that can be used by machine learning models
            to predict glycemic events.
        """
        if 'pasthistorypath' not in df.columns:
            raise ValueError("DataFrame must contain 'pasthistorypath' column")

        output = pd.DataFrame(index=df.index)
        
        # Ensure string type and handle NaNs
        paths = df['pasthistorypath'].astype(str).fillna('')

        for name, pattern in self.patterns.items():
            output[name] = paths.apply(lambda x: 1 if re.search(pattern, x) else 0)
            
        return output
