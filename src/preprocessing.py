"""
Data cleaning utilities for developer salary raw csv.
We'll use these functions in train.py/model.py and app.py
"""

import pandas as pd
import numpy as np

# constants. - it's good practice to write all constants at the top of the file
TARGET = 'ConvertedCompYearly'
SALARY_MIN = 10_000
SALARY_MAX = 500_000
SELECTED_FEATURES = ['Country', 'YearsCode', 'EdLevel', 'Employment', 'LanguageHaveWorkedWith',
                     #new features to add
                     "DevType", # developer role
                     "OrgSize", # company size
                     "RemoteWork", # remote/hybrid/ in-person
                     "WorkExp",
                     "Industry", 
                     "Age", # age band(Use ordinal 0-6 to represent)
                     "ICorPM", # individual contributor or people manager
                     "DatabaseHaveWorkedWith", # we can hv database count and split by semicolon
                     "PlatformHaveWorkedWith", # we can have platform count and split by semicolon
                     "ToolCountWork" # 
                     ]
TOP_COUNTRIES = 25
LOG_TARGET = 'log_salary'
EMPLOYMENT_KEEP = ['Full-time', 'Freelance/Self-employed']
TARGET_ENC_FEATURES = ['Country', 'DevType', 'Industry']

# mappings for features that we'll do label encoding
ED_LEVEL_ORDINAL: dict[str, int] = {
    "Primary school": 0,
    "High school": 1,
    "Some college": 2,
    "Associate's": 3,
    "Bachelor's": 4,
    "Master's": 5,
    "Professional":6,
    "Other": np.nan   
}

REMOTE_ORDINAL: dict[str, int] = {
    "In-person": 0,
    "Hybrid": 1,
    "Remote": 2,
    "Other": np.nan
}

AGE_ORDINAL: dict[str, int] = {
    "18-24 years old": 0,
    "25-34 years old": 1,
    "35-44 years old": 2,
    "45-54 years old": 3,
    "55-64 years old": 4,
    "65 years or older": 5,
    "Prefer not to say": np.nan
}

ORGSIZE_ORDINAL: dict[str, int] = {
    'Just me - I am a freelancer, sole proprietor, etc.':0,
    'Less than 20 employees': 1,
    '20 to 99 employees': 2,
    '100 to 499 employees': 3,
    '500 to 999 employees': 4,
    '1,000 to 4,999 employees': 5,
    '5,000 to 9,999 employees': 6,
    "10,000 or more employees": 7,
    "I don't know": np.nan
}


def clean_years_code(series: pd.Series) -> pd.Series: # helps enforce the output to be pandas series
    '''converts YearsCode to numeric'''
    series = series.copy()
    series = pd.to_numeric(series, errors='coerce') # helps if it meets anything that cannot converted to numeric say you got a string it will be replaced with NaN)
    return series

def clean_work_exp(series: pd.Series) -> pd.Series:
    """Converts Workexp to numeric"""
    series = pd.to_numeric(series)
    return series



def clean_education(series: pd.Series) -> pd.Series:
    '''Starndardizing EdLevel into a set of clean categories
    Then mapping these clean categories to ordinal integer'''

    mapping = {
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Bachelor's",
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "Master's",
        "Some college/university study without earning a degree": "Some college",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "High school",
        "Associate degree (A.A., A.S., etc.)": "Associate's",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional",
        "Other (please specify):": "Other",
        "Primary/elementary school":"Primary school"
    }
    toordinal = series.map(mapping).fillna("Other") 
    return toordinal.map(ED_LEVEL_ORDINAL)

def clean_employment(series: pd.Series) -> pd.Series:
    '''simplifying employment into core categories'''

    def simplify(val):
        if pd.isna(val):
            return np.nan # returns it as it is, the transformation will be done by the pipeline
        
        val = str(val)
        if 'employed' in val.lower() or 'full-time' in val.lower():
            return 'Full-time'
        elif 'Independent contractor, freelancer, or self-employed' in val.lower():
            return 'Freelance/Self-employed'
        elif 'student' in val.lower():
            return 'Student'
        else:
            return 'Other'
    
    return series.apply(simplify)

def clean_age(series: pd.Series) -> pd.Series:
    """ Mapping age bands to ordinal intergers"""
    return series.map(AGE_ORDINAL)

def clean_org_size(series: pd.Series) -> pd.Series: # enforcing output to be series
    """ mapping org-size bands to ordinal integers"""
    return series.map(ORGSIZE_ORDINAL)

def clean_icorpm(series: pd.Series) -> pd.Series:
    """ mapping IC or PM role to binary: Manager - 1, IC - 0
    then we use underscore for a f(x) within a f(x)"""
    def _map(val):
        if pd.isna(val):
            return np.nan
        v = str(val).lower()
        if "manager" in v or "lead" in v:
            return 1
        return 0
    return series.apply(_map)

def clean_remote_work(series: pd.Series) -> pd.Series:
    """mapping remote work values to ordinal integers"""
    mapping = {
        "Remote" : "Remote",
        "Hybrid (some in-person, leans heavy to flexibility)" :"Hybrid",
        "Hybrid (some remote, leans heavy to in-person)" : "Hybrid",
        "In-person": "In-person",
        "Your choice (very flexible, you can come in when you want or just as needed)": "Other"
    }
    incomplete = series.map(mapping).fillna("Other")
    return incomplete.map(REMOTE_ORDINAL)



def count_languages(series: pd.Series) -> pd.Series:
    """ 
        convert comma separated language list into a count.

        Example: ' Bash/Shell (all shells);Dart;SQL' -> 3

        we apply val.split(';')
        -> val => [Python, JavaScript, SQL, HTML]
        retuen len(val) # 4
    """
    def _count(val): # this starts with an underscore good naming convetion
        if pd.isna(val) or val == '':
            return np.nan
        return len(str(val).split(";"))
    
    return series.apply(_count) # we apply the inner f(x) to our series


def group_rare_countries(series: pd.Series, top_n: int = TOP_COUNTRIES) -> pd.Series:
    """
        Keeping only the top N most common countries (default = 15). Replace all others with 'Other'.
    """
    top_countries = series.value_counts().head(top_n).index.tolist() # the index helps us pick only the indexing leaving behind the count
    return series.apply(lambda x : x if x in top_countries else 'Other') # lambda helps create a temporary f(x)

def clean_industry(series: pd.Series) -> pd.Series:
    " Keeping only the top 10 industries and replacing all others with 'other'"
    top = series.value_counts().head(10).index.tolist()
    return series.apply(lambda x: x if x in top else 'Other')

def clean_dev_type(series: pd.Series) -> pd.Series:
    """ picking the primary developer role"""
    def _primary(val):
        if pd.isna(val): # allows us to check if an object is empty
            return "Other"
        low = str(val).lower()
        if "full-stuck" in low:
            return "Full-stack"
        if "back-end" in low:
            return "Back-end"
        if "front-end" in low:
            return "Front-end"
        if "software" in low:
            return "Software"
        if "desktop" in low:
            return "Desktop"
        if "embedded" in low or "hardware" in low:
            return "Embedded/Hardware"
        if "devops" in low or "cloud" in low or "site reliability" in low:
            return "DevOps/Cloud"
        if "mobile" in low:
            return "Mobile"
        if "data scientist" in low or "machine learning" in low or "ml" in low or "data engineering" in low or "data analyst" in low:
            return "Data/ML"
        if "manager" in low or "executive" in low or "director" in low:
            return "Management"
        return "Other"
    return series.apply(_primary)

def count_items(series: pd.Series) -> pd.Series:
    """For counting semi-colon separated items in a column, we are 
    going to return  if blank"""
    def _count(val):
        if pd.isna(val) or val=="":
            return np.nan
        return len(str(val).split(";"))
    
    return series.apply(_count)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        why these features help:
            - YearsCode_sq : diminishing returns on salary growth per year
            - WorkExp_sq : same as above
            - Exp_ratio: what fraction of programming/coding time was professional
            - Tech_breadth : total tools known = shows seniority. 
        """
        df['yearsCode_sq'] = df['YearsCode'] ** 2 # we square to help us have a bigger difference btwn the years of experience
        df['WorkExp_sq'] = df['WorkExp'] ** 2
        df['Exp_ratio'] = df['WorkExp']/ (df['YearsCode'] + 1) # we add the 1 to avoid the division by zero error
        df['Tech_breadth'] = (df['LanguageCount'] + df['DatabaseCount'] + df['PlatformCount']).fillna(0)
        
        return df


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
        Loading the raw Stack Overflow survey csv and return a clean DF.
        The DF will be ready for the scikit-learn pipeline.

        parameters to pass :filepath to the raw csv file.

        returns pd.DataFrame (DF with features + target column)

        Also note its good to have the print statement bekoz we are not on jupyter
    """
    # step 0 - loading the data
    df = pd.read_csv(filepath, low_memory=False)
    print(f" This is the raw shape: {df.shape} \n")

    # step 1 - Keep only rows with a valid salary and do some filtering
    df = df.dropna(subset=[TARGET]) # the dropna drops the entire row with reference to target col, the subsets ask with columns to have
    df = df[df[TARGET].between(SALARY_MIN, SALARY_MAX)] # to see a list of boolean values
    print(f"Shape after salary filter: {df.shape}")

    # step 1.5 - Log transforming salary
    df[LOG_TARGET] = np.log1p(df[TARGET])

    # We are transforming our target variable into a log using log1p
    # log1p helps handle the divide by zero error
 # this is the log transformation of the target variable, we will use this in our model training and prediction, but we will keep the original salary for the app display and user input


    # step 2: check whether the feature and target columns exist in the df
    cols_needed = SELECTED_FEATURES + ['log_salary'] # we write the target in [] to cast it into a list
    cols_available = [c for c in cols_needed if c in df.columns] # df.columns columns is a key word

    missing_cols = set(cols_needed) - set(cols_available) # we convert the list to a set
    if missing_cols:
        print(f" You dont have column(s): {missing_cols} in your dataset")

    df = df[cols_available].copy()
    print(f" selected {len(cols_available)} columns, expecting 6 columns")

    # step 3: Cleaning individual columns by function call

    if 'YearsCode' in df.columns: # this is a jsut a check but necessary
        df['YearsCode'] = clean_years_code(df['YearsCode'])

    if 'WorkExp' in df.columns:
        df['WorkExp'] = clean_work_exp(df['WorkExp'])


    if 'EdLevel' in df.columns:
        df['EdLevel'] = clean_education(df['EdLevel'])

    if 'Employment' in df.columns:
        df['Employment'] = clean_employment(df['Employment'])

    if 'Age' in df.columns:
        df['Age'] = clean_age(df['Age'])

    if 'OrgSize' in df.columns:
        df['OrgSize'] = clean_org_size(df['OrgSize'])

    if 'ICorPM' in df.columns:
        df['ICorPM'] = clean_icorpm(df['ICorPM'])

    if 'RemoteWork' in df.columns:
        df['RemoteWork'] = clean_remote_work(df['RemoteWork'])

    if 'DevType' in df.columns:
        df['DevType'] = clean_dev_type(df['DevType'])
    
    if 'LanguageHaveWorkedWith' in df.columns:
        df['LanguageCount'] = count_languages(df['LanguageHaveWorkedWith'])
        df = df.drop(columns=['LanguageHaveWorkedWith'])

    if 'DatabaseHaveWorkedWith' in df.columns:
        df['DatabaseCount'] = count_items(df['DatabaseHaveWorkedWith'])
        df = df.drop(columns=['DatabaseHaveWorkedWith'])

    if 'PlatformHaveWorkedWith' in df.columns:
        df['PlatformCount'] = count_items(df['PlatformHaveWorkedWith'])
        df = df.drop(columns=['PlatformHaveWorkedWith'])

    if 'ToolCountWork' in df.columns:
        df['ToolCountWork'] = pd.to_numeric(df['ToolCountWork'], errors ='coerce')

    if 'Country' in df.columns:
        df['Country'] = group_rare_countries(df['Country'])


    # step 4: filtering employment
    if 'Employment' in df.columns:
        before = len(df) # just to note the length of the df before filter
        df = df[df['Employment'].isin(EMPLOYMENT_KEEP)] # we shall have boolean TF
        df['Employment'] = (df['Employment']== 'Full-time').astype(int) # we you cast True =1

        print(f"Employment filter: {before} -> {len(df)} rows"
              f"We have kept only Full-time and Freelance")

    # step 5: Add interaction features (Polynimial features)
    df = df.dropna(how='all')


    # step 6: drop rows where all features are NaN (Handling this edge case)
    df = df.dropna(how='all') # this check if the user only filled the salary alone

    print(f"Clean data shape: {df.shape}")
    print(f" Missing values per column: \n {df.isna().sum().to_string()}  \n")

    return df

def get_feature_columns(df: pd.DataFrame) -> tuple[list, list]:
    """
        returns (categorical columns, numeric columns) from the cleaned DF, excluding the target
        variable
    """
    non_target = [c for c in df.columns if c != LOG_TARGET]
    target_enc = [c for c in TARGET_ENC_FEATURES if c in non_target]

    # cat_cols = df[non_target].select_dtypes(include=['object', 'category']).columns.to_list() - 
    #we remove this code bekoz you you find that all the other categorical cols were converted.
     

    num_cols = df[non_target].select_dtypes(include=['number']).columns.to_list()

    return target_enc, num_cols
