import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype

import os 

def one_hot_encode_columns(df, columns):
    """
    One-hot encode specified columns in a DataFrame.

    :param df: The DataFrame to process.
    :param columns: A list of column names to be one-hot encoded.
    :return: The modified DataFrame and a list of new column indices.
    """
    original_columns = set(df.columns)
    for column in columns:
        encoded = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, encoded], axis=1).drop(column, axis=1)
    new_columns = set(df.columns) - original_columns
    new_column_indices = [df.columns.get_loc(c) for c in new_columns]
    return df, new_column_indices

def scale_columns(df, scaler, columns):
    """
    Scale specified columns using the provided scaler.

    :param df: The DataFrame to process.
    :param scaler: Scaler instance (e.g., MinMaxScaler).
    :param columns: A list of column names to be scaled.
    :return: The modified DataFrame.
    """
    df[columns] = scaler.fit_transform(df[columns])
    return df

def most_common_ratio(series):
    """
    Calculate the ratio of the most common value in a Series.

    :param series: Pandas Series to analyze.
    :return: Ratio of the most common value.
    """
    return series.value_counts(normalize=True).iloc[0]

def remove_column_with_high_ratio(df, exclude_columns):
    """
    Remove columns with a high ratio of a single value.

    :param df: The DataFrame to process.
    :param exclude_columns: A list of column names to be excluded from processing.
    :return: The modified DataFrame.
    """
    threshold = 0.98
    cols_to_drop = [col for col in df.columns if most_common_ratio(df[col]) > threshold and col not in exclude_columns]
    return df.drop(columns=cols_to_drop)

def creating_indicator_variables(df):
    """
    Create indicator variables for columns with missing values.

    :param df: The DataFrame to process.
    :return: The modified DataFrame.
    """
    indicator_dict = {}
    for col in df.columns:
        if df[col].isna().any():
            indicator_col_name = f"{col}_present"
            indicator_dict[indicator_col_name] = df[col].notna().astype(int)
    indicator_df = pd.DataFrame(indicator_dict)
    return pd.concat([df, indicator_df], axis=1)

def custom_imputation(df):
    """
    Apply custom imputation to DataFrame columns based on the percentage of missing values.

    :param df: The DataFrame to process.
    :return: The modified DataFrame.
    """
    for col in df.columns:
        if df[col].isna().mean() > 0.40:
            df[col].fillna(0, inplace=True)
        elif is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
    return df

def remove_highly_correlated_column(df, exclude_columns):
    """
    Remove highly correlated columns from the DataFrame.

    :param df: The DataFrame to process.
    :param exclude_columns: A list of column names to be excluded from processing.
    :return: The modified DataFrame.
    """
    threshold = 0.99
    corr_matrix = df.drop(columns=exclude_columns).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold) and column not in exclude_columns]
    return df.drop(columns=to_drop)

def remove_column_with_zero_variance(df):
    """
    Remove columns with zero variance from the DataFrame.

    :param df: The DataFrame to process.
    :return: The modified DataFrame.
    """
    return df.loc[:, df.var() != 0]

def ensure_column_are_numeric(df):
    """
    Ensure all columns in the DataFrame are of numeric type.

    :param df: The DataFrame to process.
    :return: The modified DataFrame.
    """
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def impute_with_median(df):
    """
    Impute missing values in the DataFrame with the median.

    :param df: The DataFrame to process.
    :return: The modified DataFrame.
    """
    non_empty_columns = df.columns[df.notna().any()].tolist()
    imputer = SimpleImputer(strategy='median')
    df = imputer.fit_transform(df)
    return pd.DataFrame(df, columns=non_empty_columns)

def impute_with_mean(df):
    """
    Impute missing values in the DataFrame with the median.

    :param df: The DataFrame to process.
    :return: The modified DataFrame.
    """
    non_empty_columns = df.columns[df.notna().any()].tolist()
    imputer = SimpleImputer(strategy='mean')
    df = imputer.fit_transform(df)
    return pd.DataFrame(df, columns=non_empty_columns)

def impute_with_zero(df):
    """
    Impute missing values in the DataFrame with the median.

    :param df: The DataFrame to process.
    :return: The modified DataFrame.
    """
    non_empty_columns = df.columns[df.notna().any()].tolist()
    df = df[non_empty_columns]
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    df = imputer.fit_transform(df)
    return pd.DataFrame(df, columns=non_empty_columns)

def determine_columns_to_scale(df, columns_to_one_hot_encode):
    """
    Determine which columns in the DataFrame should be scaled.

    :param df: The DataFrame to process.
    :param columns_to_one_hot_encode: A list of column names that should not be scaled.
    :return: A list of column names to be scaled.
    """
    all_columns = set(df.columns)
    non_scale_columns = set(columns_to_one_hot_encode)
    return list(all_columns - non_scale_columns)

def feature_engineering_C_part1(df, target_columns):
    """
    Perform feature engineering on the DataFrame for part C-1.

    :param df: The DataFrame to process.
    :param target_columns: A list of target column names.
    :return: The modified DataFrame.
    """
    columns_to_one_hot_encode = ["architectural_archetype", "stories", "soil_class", "seismic_zone", "connection_system", 'Story', 'Direction', 'Wall']
    columns_to_scale = ['L cm', 'xi cm', 'yi cm', 'D+0.25L', 'Story Area']
    
    target_data = df[target_columns]
    df = df.drop(columns=target_columns)
    df,new_column_indices = one_hot_encode_columns(df, columns_to_one_hot_encode)
    ensure_column_are_numeric(df)
    columns_to_scale
    scale_columns(df, MinMaxScaler(), columns_to_scale)
    df = remove_column_with_zero_variance(df)
    df = pd.concat([df, target_data], axis=1)
    return df

def feature_engineering2(df, target_columns):
    """
    Perform feature engineering on the DataFrame for part C-2 and D.

    :param df: The DataFrame to process.
    :param target_columns: A list of target column names.
    :return: The modified DataFrame.
    """
    columns_to_one_hot_encode = ["architectural_archetype", "stories", "soil_class", "seismic_zone", "connection_system"]
    target_data = df[target_columns]
    df = df.drop(columns=target_columns)
    
    # One-hot encode specified columns
    df, new_column_indices = one_hot_encode_columns(df, columns_to_one_hot_encode)
    
    df = impute_with_zero(df)
    ensure_column_are_numeric(df)
    df = scale_columns(df, MinMaxScaler(), determine_columns_to_scale(df, columns_to_one_hot_encode))
    
    df = remove_column_with_zero_variance(df)
    exclude_columns = ["connection_system_ATS", "connection_system_HD"]
    df = remove_highly_correlated_column(df, exclude_columns)
    df_dropped = remove_column_with_high_ratio(df, exclude_columns)
    df = pd.concat([df_dropped, target_data], axis=1)
    
    return df

import pandas as pd

def read_and_process_data(file_path, target_columns, feature_engineering_func):
    """
    Read data from a CSV file, apply feature engineering, and return the processed DataFrame.

    :param file_path: Path to the CSV file.
    :param target_columns: List of target columns for feature engineering.
    :param feature_engineering_func: Function to be used for feature engineering.
    :return: Processed DataFrame.
    """
    df = pd.read_csv(file_path, low_memory=False)
    return feature_engineering_func(df, target_columns)

def save_processed_data(df, file_path):
    """
    Save the processed DataFrame to a CSV file.

    :param df: DataFrame to be saved.
    :param file_path: Path where the DataFrame will be saved.
    """
    df.to_csv(file_path, index=False)



if __name__ == "__main__":
    # Paths for source data and processed data
    path = './Files/Before_Feature_Engineering'
    path_FE = 'Files/After_Feature_Engineering/'


    #If the output directory does not exist , we create it 
    if not os.path.exists(path_FE):
        print("Creating the folder :"+path_FE)
        os.makedirs(path_FE)

    # Feature Engineering for C part 1
    file_path_C_part1 = path + '/data_C_part1.csv'
    target_column_C = ["Nail spacing [cm]", "Number sheathing panels", "Number end studs", "Total number studs", "HoldDown Model / ATS"]
    df_C_part1 = read_and_process_data(file_path_C_part1, target_column_C, feature_engineering_C_part1)
    save_processed_data(df_C_part1, path_FE + 'data_C_part1_FE.csv')

    # Feature Engineering for C part 2
    file_path_C_part2 = path + '/data_C_part2.csv'
    target_column_C_2 = ['Tx(s)', 'Ty(s)']
    df_C_part2 = read_and_process_data(file_path_C_part2, target_column_C_2, feature_engineering2)
    save_processed_data(df_C_part2, path_FE + 'data_C_part2_FE.csv')

    # Feature Engineering for D
    file_path_D = path + '/data_D.csv'
    target_column_D = ['Ωx', 'Ωy', 'µx', 'µy', 'CMR', 'SSF', 'ACMR', 'IO-ln θ', 'IO-β', 'LS-ln θ', 'LS-β', 'CP-ln θ', 'CP-β']
    df_D = read_and_process_data(file_path_D, target_column_D, feature_engineering2)
    save_processed_data(df_D, path_FE + 'data_D_FE.csv')

