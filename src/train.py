""""
    end to end training script
    
    usage -> in terminal, after activating your venv, python scr/model.py

    outputs -> saved pipeline as .pkl,
                cleaned data, .csv
                the plot, .png

"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import joblib # Helps with saving the pipeline (.joblib or .peakle)

from preprocessing import load_and_clean, get_feature_columns, LOG_TARGET
from evaluate import evaluate_model, plot_predictions, print_observations

""" we do some configurations so that a user maybe able to change anything from the top of the file"""

# add scr/ to path
sys.path.insert(0, os.path.dirname(__file__)) # helps the path to just change dynamically

# configuration
RAW_DATA_PATH = 'data/raw/developer-survey-2025.csv'
PROCESSED_DATA = 'data/cleaned/processed-data.csv'
MODEL_OUTPUT_PATH = 'models/salary_pipeline_v2.pkl'


RANDOM_STATE = 42
TEST_SIZE = 0.2

XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth' : 5,
    'learning_rate' : 0.05,
    'subsample' : 0.8,
    'colsample_bytre' : 0.8,
    'random_state' : RANDOM_STATE,
    'tree_method' : 'hist',
    'verbosity' : 0 ,
    'reg_alpha': 0.05, # L1 regularization, feature selection.
    'reg_lambda': 1.0, # L2 regularization, weight shrinkage.
}

def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    """
        1. num pipeline
        2. cat pipeline
        3. column transformer
    """
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', TargetEncoder(smooth='auto', target_type='continuous', random_state=RANDOM_STATE))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, num_cols),
        ('cat', categorical_pipeline, cat_cols)
    ])

    return preprocessor

def build_pipeline(cat_cols: list, num_cols: list) -> Pipeline :
    """
    Combine preprocessor + xgboost model into one sklearn pipeline
    """
    preprocessor = build_preprocessor(cat_cols, num_cols)
    model = XGBRegressor(**XGBOOST_PARAMS)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
 
    return pipeline

# our main function which we shall run
def main():
    # 1. Load and clean data
    df = load_and_clean(RAW_DATA_PATH)
    df.to_csv(PROCESSED_DATA, index=False)
    print(f'csv saved to {PROCESSED_DATA} \n')

    #2. separating features from the target
    X = df.drop(columns=[LOG_TARGET])
    y = df[LOG_TARGET]

    cat_cols, num_cols = get_feature_columns(df)
    print(f'we have cat cols {cat_cols} and num cols {num_cols} \n')

    #3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f'We have split the data {TEST_SIZE * 100}% to testing the rest to training')

    #4. Building and training the pipeline
    print('Building pipeline ...')
    pipeline = build_pipeline(cat_cols, num_cols)

    print('Training XGBoost model...\n')
    pipeline.fit(X_train, y_train)
    print('Training complete. \n')

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)

    print("\ntesting metrics\n")
    test_metrics = evaluate_model(y_test, y_pred, title='test set performance')
    print_observations(test_metrics)

    print("\ntraining metrics\n")
    train_metrics = evaluate_model(y_train, y_pred_train, title='training set performance')
    print_observations(train_metrics)

    # plot_predictions(y_test.values, y_pred, 'data/predictions_plot.png')

    #6. Save the pipeline
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}\n")

    '''
    # 7. Predicting one example
    print('Sample prediction: ')
    sample = pd.DataFrame([
        {
            'Country': 'Kenya',
            'YearsCode': 5.0,
            'EdLevel': "Bachelor's",
            'Employment': "Full-time",
            'LanguageCount': 3
        }
    ])
    pred = pipeline.predict(sample)[0]
    print(f"Input: {sample.to_dict(orient='records')[0]}")
    print(f"predicted salary: ${pred:,.0f}")
    '''
    print('\n Training script complete. \n')


# y_test.values hepls us to convert it from panda series, plotting execpts a numpy array
# we also add the path save path durin the evaluate
# orient returns a list like output of records


if __name__ == '__main__':
    main()

# regularization, hyperparameter tuning 
# 



