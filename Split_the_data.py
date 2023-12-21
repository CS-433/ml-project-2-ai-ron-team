
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_file_path(base_path, suffix, file_type, name_plus):
    """
    Create a file path for saving datasets.

    :param base_path: The base directory where the file will be saved.
    :param suffix: Suffix for the file name (before file extension).
    :param file_type: The type of file, e.g., 'X_train', 'X_test', etc.
    :param name_plus: Additional identifier to be appended to the file name.
    :return: Constructed file path as a string.
    """
    return f"{base_path}/{file_type}_{name_plus}{suffix}.csv"

def split_and_save_data(file_path, target_columns, name_plus, path_to):
    """
    Split the dataset into training and testing sets and save them as CSV files.

    :param file_path: Path to the source CSV file.
    :param target_columns: List of target column names.
    :param name_plus: Additional identifier for the output file names.
    :param path_to: Base path where the split files will be saved.
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    X = df.drop(columns=target_columns)
    Y = df[target_columns]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Saving split data to CSV
    for dataset, file_type in zip([X_train, X_test, Y_train, Y_test], ['X_train', 'X_test', 'Y_train', 'Y_test']):
        file_path = create_file_path(path_to, suffix='', file_type=file_type, name_plus=name_plus)
        dataset.to_csv(file_path, index=False)

def handle_splitting(path_from, path_to, file_name, target_columns, name_plus):
    """
    Handle the data splitting process for different datasets.

    :param path_from: Directory from where the source CSV file is read.
    :param path_to: Directory where the split files will be saved.
    :param file_name: Name of the source CSV file (without extension).
    :param target_columns: List of target column names for splitting.
    :param name_plus: Additional identifier for the output file names.
    """
    file_path = f"{path_from}/{file_name}.csv"
    split_and_save_data(file_path, target_columns, name_plus, path_to)


if __name__ == "__main__":
    #orchestrate the data splitting process for different datasets.#

    path_from_before_FE = './Files/Before_Feature_Engineering'
    path_to_before_FE = './Files/Before_Feature_Engineering/Split'
    path_from_after_FE = './Files/After_Feature_Engineering'
    path_to_after_FE = './Files/After_Feature_Engineering/Split'

    target_columns_C = ["Nail spacing [cm]", "Number sheathing panels", "Number end studs", "Total number studs", "HoldDown Model / ATS"]
    target_columns_C_2 = ['Tx(s)', 'Ty(s)']
    target_columns_D = ['Ωx', 'Ωy', 'µx', 'µy', 'CMR', 'SSF', 'ACMR', 'IO-ln θ', 'IO-β', 'LS-ln θ', 'LS-β', 'CP-ln θ', 'CP-β']


    #If the output directories does not exist , we create them 
    if not os.path.exists(path_to_before_FE):
        print("Creating the folder :"+path_to_before_FE)
        os.makedirs(path_to_before_FE)


    if not os.path.exists(path_to_after_FE):
        print("Creating the folder :"+path_to_after_FE)
        os.makedirs(path_to_after_FE)
        




    # Splitting data before Feature Engineering
    handle_splitting(path_from_before_FE, path_to_before_FE, 'data_C_part1', target_columns_C, 'C_part1')
    handle_splitting(path_from_before_FE, path_to_before_FE, 'data_C_part2', target_columns_C_2, 'C_part2')
    handle_splitting(path_from_before_FE, path_to_before_FE, 'data_D', target_columns_D, 'D')

    # Splitting data after Feature Engineering
    handle_splitting(path_from_after_FE, path_to_after_FE, 'data_C_part1_FE', target_columns_C, 'C_part1_FE')
    handle_splitting(path_from_after_FE, path_to_after_FE, 'data_C_part2_FE', target_columns_C_2, 'C_part2_FE')
    handle_splitting(path_from_after_FE, path_to_after_FE, 'data_D_FE', target_columns_D, 'D_FE')