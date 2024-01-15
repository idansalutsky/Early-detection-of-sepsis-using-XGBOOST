#from preprocess import *
import numpy as np
import joblib
import sys, os, csv
import pandas as pd
import xgboost as xgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

def extract_df_comp(Folder):
    data_frames = []

    for filename in os.listdir(Folder):
          with open(os.path.join(Folder, filename), 'r') as f:
              reader = csv.reader(f, delimiter='|')
              data = [row for row in reader]
              headers = data[0]
              df = pd.DataFrame(data[1:], columns=headers)
              df['index'] = os.path.join(filename)
              if (df['SepsisLabel'] == '1').any():
                  index_of_first_1 = (df['SepsisLabel'] == '1').idxmax()
                  df['label'] = 1
                  df = df.iloc[:index_of_first_1+1]
              else:
                  df['label'] = 0
              data_frames.append(df)

    df_all = pd.concat(data_frames, ignore_index=True)
    df_all.loc[:, df_all.columns != 'index'] = df_all.loc[:, df_all.columns != 'index'].apply(pd.to_numeric, errors='coerce')
    return df_all


def grouped_comp(df_to_groupby):
    to_stats = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
          'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
          'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
          'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
          'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
          'Fibrinogen', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'ICULOS', 'label']


    # Group by index column and apply multiple aggregation functions
    agg_dict = {'HR': ['mean', 'std'], 
                'O2Sat': ['mean', 'std'], 
                'Temp': ['mean', 'std'], 
                'SBP': ['mean', 'std'], 
                'MAP': ['mean',  'std'], 
                'DBP': ['mean', 'std'], 
                'Resp': ['mean', 'std'], 
                'EtCO2': ['mean', 'std'], 

                'BaseExcess': ['mean', 'std'], 
                'HCO3': ['mean', 'std'], 
                'FiO2': ['mean', 'std'], 
                'pH': ['mean', 'std'], 
                'PaCO2': ['mean', 'std'], 
                'SaO2': ['mean', 'std'], 
                'AST': ['mean', 'std'], 
                'BUN': ['mean', 'std'], 

                'Alkalinephos': ['mean','std'], 
                'Calcium': ['mean', 'std'], 
                'Chloride': ['mean', 'std'], 
                'Creatinine': ['mean', 'std'], 
                'Bilirubin_direct': ['mean', 'std'], 

                'Glucose': ['mean', 'std'], 
                'Lactate': ['mean',  'std'], 
                'Magnesium': ['mean', 'std'], 
                'Phosphate': ['mean', 'std'], 
                'Potassium': ['mean', 'std'], 

                'Bilirubin_total': ['mean', 'std'], 
                'TroponinI': ['mean', 'std'], 
                'Hct': ['mean', 'std'], 
                'Hgb': ['mean', 'std'], 
                'PTT': ['mean', 'std'], 
                'WBC': ['mean', 'std'], 

                'Fibrinogen': ['mean', 'std'], 
                'Platelets': ['mean', 'std'], 
                'Age': ['mean'], 
                'Gender': ['mean'],  
                'HospAdmTime': ['mean'], 
                'ICULOS': ['mean', 'std'], 
                'label': ['mean'], 
                }

    grouped_df = df_to_groupby.groupby('index')[to_stats].agg(agg_dict)

    # Flatten column names in the resulting DataFrame
    grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]

    # Reset the index to make the groupby column a regular column
    grouped_df = grouped_df.reset_index()
    return grouped_df

def impute_knn_comp(df_to_impute):

    missing_cols = [col for col in df_to_impute.columns if df_to_impute[col].isnull().any()]
    missing_cols_idx = [df_to_impute.columns.get_loc(col) for col in missing_cols]

    imputer = KNNImputer(n_neighbors=3)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_to_impute), columns=df_to_impute.columns)
    return df_imputed
    


# Define the path to the patient tables folder as the first argument in the command line
folder_path = sys.argv[1]

test_df = extract_df_comp(folder_path)
test_df_grouped = grouped_comp(test_df)
best_cols = ['HR_mean', 'O2Sat_mean', 'Temp_mean', 'MAP_mean',  'Resp_mean','Chloride_mean', 'Lactate_mean', 'Hct_mean', 'Hgb_mean',
      'WBC_mean', 'Platelets_std', 'HospAdmTime_mean', 'ICULOS_mean',
      'ICULOS_std']
test_grouped_copy = impute_knn_comp(test_df_grouped[best_cols])


xgb_loaded = joblib.load('comp_xgb_model.joblib')
dcomp_test = xgb.DMatrix(test_grouped_copy)
y_pred = xgb_loaded.predict(dcomp_test)
binary_predictions = np.where(y_pred > 0.37, 1, 0)

new_df = pd.DataFrame({'id': test_df_grouped['index'], 'prediction': binary_predictions})
new_df.to_csv('prediction.csv', index=False)
