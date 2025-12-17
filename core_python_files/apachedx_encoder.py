import re
import pandas as pd

class ApacheDxEncoder:
    """
    Encodes patient admission diagnosis data into binary features relevant to glycemic control.
    
    This class uses robust regex pattern matching to identify high-impact clinical conditions
    (e.g., Sepsis, Stroke, DKA) from the 'apacheadmissiondx' column of the eICU 'patient' table.
    
    Attributes:
        patterns (dict): A dictionary mapping feature names (e.g., 'admit_sepsis') to 
                         regex patterns used for identification.
    """
    def __init__(self):
        # Robust regex patterns for high-impact admission diagnoses
        self.patterns = {
            # Sepsis: Covers sepsis, septic shock, infections, urosepsis, bacteremia
            'admit_sepsis': r'(?i)(sepsis|septic|infection|urosepsis|bacteremia)',
            
            # Stroke/CVA: Covers strokes, infarcts, hemorrhages (intracranial/subarachnoid)
            'admit_stroke': r'(?i)(stroke|CVA|cerebrovascular|infarct|hemorrhage.*(brain|intracranial|subarachnoid|cerebral)|bleed.*(brain|head))',
            
            # Cardiac Arrest: Explicit cardiac arrest
            'admit_cardiac_arrest': r'(?i)(arrest.*cardiac|cardiac.*arrest)',
            
            # DKA/Hyperglycemia: Diabetic Ketoacidosis, Hyperglycemic Hyperosmolar State, Uncontrolled Diabetes
            'admit_dka': r'(?i)(ketoacidosis|DKA|hyperglycem|diabet.*uncontrolled|diabet.*coma)',
            
            # Hypoglycemia: Explicit hypoglycemia
            'admit_hypoglycemia': r'(?i)(hypoglycem)',
            
            # Trauma: Trauma, injuries, fractures (excluding pathological)
            'admit_trauma': r'(?i)(trauma|injury|fracture|accident|contusion|laceration)',
            
            # Surgery: Post-op, surgery for X, transplants, grafts, resections
            'admit_surgery': r'(?i)(operative|surgery|post-op|graft|transplant|resection|ectomy|repair)',
            
            # Respiratory Failure: Respiratory arrest, failure, dependence on ventilator, ARDS
            'admit_resp_failure': r'(?i)(respiratory\s+failure|arrest,\s+respiratory|ventilator|intubated|ARDS|adult\s+respiratory\s+distress)'
        }

    def transform(self, df):
        """
        Transforms the 'apacheadmissiondx' column into 8 binary features.
        
        Input:
            df (pd.DataFrame): A DataFrame containing the 'apacheadmissiondx' column.
                               This data comes from the 'patient' table via 
                               `DataExtractor.get_static_data`.
                               
        Actions:
            1. Checks if 'apacheadmissiondx' exists in the DataFrame.
            2. Converts the column to string and handles missing values.
            3. Iterates through each defined feature pattern.
            4. Applies regex search to each row's diagnosis string.
            5. Creates a binary column (1 if match found, 0 otherwise) for each feature.
            
        Output:
            pd.DataFrame: A new DataFrame with the same index as input, containing 
                          8 binary columns (e.g., 'admit_sepsis', 'admit_dka').
                          Returns an empty DataFrame if the input column is missing.
                          
        Purpose:
            To extract critical admission diagnoses that significantly impact glycemic 
            control (stress hyperglycemia or hypoglycemia risk) from free-text diagnosis fields.
        """
        if 'apacheadmissiondx' not in df.columns:
            return pd.DataFrame()
            
        output = pd.DataFrame(index=df.index)
        
        # Ensure string type and handle NaNs
        dx_series = df['apacheadmissiondx'].astype(str).fillna('')

        for name, pattern in self.patterns.items():
            output[name] = dx_series.apply(lambda x: 1 if re.search(pattern, x) else 0)
            
        return output
