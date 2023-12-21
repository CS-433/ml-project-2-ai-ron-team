import pandas as pd
import numpy as np

import os #Used to create folders 

#Constants
starting_row_informationB = 14
size_columns_informationA = 5
row_Tx_Ty_values = 12
columns_D = ['Ωx', 'Ωy', 'µx', 'µy', 'CMR', 'SSF', 'ACMR', 'IO-ln θ','IO-β',
           'LS-ln θ','LS-β', 'CP-ln θ','CP-β']

def parse_header(header):
    """
    Parse the header string to extract information.

    :param header: Header string containing delimited information.
    :return: Dictionary with extracted information.
    """
    parts = header.split('_')
    extracted_info = {
        "architectural_archetype": parts[0],
        "stories": int(parts[1]),
        "soil_class": parts[4],
        "seismic_zone": int(parts[6]),
        "connection_system": parts[8]
    }
    return extracted_info

def fill_values_based_on_key(data, key_column_index, value_row_index, finishing_row_informationB):
    """
    Fill values in a column based on the last non-NaN value in another column.

    :param data: DataFrame containing the data.
    :param key_column_index: Index of the key column.
    :param value_row_index: Starting row index for filling values.
    :param finishing_row_informationB: Ending row index for filling values.
    """
    last_valid_key = None
    for i in range(value_row_index, finishing_row_informationB+1):
        key_value = data.iat[i, key_column_index]
        if pd.notna(key_value):
            last_valid_key = key_value
        if pd.isna(data.iat[i, key_column_index]):
            data.iat[i, key_column_index] = last_valid_key

def filling_values(data, starting_row_informationB, finishing_row_informationB):
    """
    Fill missing values for story and direction columns in the dataset.

    :param data: DataFrame containing the data.
    :param starting_row_informationB: Starting row index for filling values.
    :param finishing_row_informationB: Ending row index for filling values.
    """
    story_index = 3
    direction_index = 4
    fill_values_based_on_key(data, story_index, starting_row_informationB, finishing_row_informationB)
    fill_values_based_on_key(data, direction_index, starting_row_informationB, finishing_row_informationB)

def find_performance_using_header(data_D, header):
    """
    Find performance data using the header information.

    :param data_D: DataFrame containing performance data.
    :param header: Header string to match in the performance data.
    :return: List of performance data corresponding to the header.
    """
    starting_column_D = 2
    row_header = data_D[data_D.iloc[:, starting_column_D] == header].index[0]
    relevant_columns = [1, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 16, 17]
    relevant_columns_D = [col + starting_column_D for col in relevant_columns]
    row_header_D = data_D.iloc[row_header, relevant_columns_D].tolist()
    return row_header_D

def get_file_information(files, path):
    """
    Get information about the files to be processed.

    :param files: List of file names.
    :param path: Base path where files are located.
    :return: List of tuples containing file paths and sheet indices.
    """
    file_information = []
    for file in files:
        xls = pd.ExcelFile(path + file)
        sheet_names = xls.sheet_names
        file_information.extend([(path + file, sheet_index) for sheet_index in range(len(sheet_names))])
    return file_information

def process_files(file_information, process_function, data_D):
    """
    Process files using a specified function.

    :param file_information: List of tuples with file paths and sheet indices.
    :param process_function: Function to process each file.
    :param data_D: DataFrame containing additional data needed for processing.
    :return: List of results from processing each file.
    """
    results = []
    for file_path, sheet_index in file_information:
        result = process_function(file_path, sheet_index, data_D)
        results.append(result)
    return results

def merge_dataframes1(dataframes, column_order=None):
    """
    Merge a list of DataFrames into a single DataFrame.

    :param dataframes: List of DataFrames to merge.
    :param column_order: Optional list of column names to order the columns in the merged DataFrame.
    :return: Merged DataFrame.
    """
    merged_df = pd.concat(dataframes, axis=0, ignore_index=True)
    if column_order:
        merged_df = merged_df[column_order]
    return merged_df

def merge_dataframes2(resultsFinal, column_order=None):
    dataframes_to_merge = resultsFinal
    merged_df = dataframes_to_merge[0]
    for df in dataframes_to_merge[1:]:
        merged_df = pd.concat([merged_df, df], axis=0, ignore_index=True)
    if column_order:
        merged_df = merged_df[column_order]
    return merged_df

def save_to_csv(df, file_path):
    """
    Save a DataFrame to a CSV file.

    :param df: DataFrame to save.
    :param file_path: File path for the output CSV file.
    """
    df.to_csv(file_path, index=False)

def parse_header_data(data, nbr_building):
    """
    Parse header data and create a DataFrame.
    Extract Type A Informatio

    :param data: DataFrame containing the headers.
    :param nbr_building: Number of buildings (headers) to process.
    :return: DataFrame with parsed header data.
    """
    columns = ["architectural_archetype", "stories", "soil_class", "seismic_zone", "connection_system"]
    parsed_data = []
    new_table = []
    for i in range(1, nbr_building+1):
        header = data[i][1]
        parsed_data.append([parse_header(header), header])
        
    for item in parsed_data:
        row = [item[0][col] for col in columns]
        new_table.append(row + [item[1]])

    return pd.DataFrame(new_table, columns=columns + ["header"])

def compute_nbr_building(data):
    """
    Compute the number of buildings based on named columns in the DataFrame.

    :param data: DataFrame to analyze.
    :return: Number of buildings.
    """
    return len([col for col in data.columns if not 'Unnamed' in str(col)])

def load_excel_data(file_path, sheet_name):
    """
    Load data from an Excel file.

    :param file_path: Path to the Excel file.
    :param sheet_name: Name of the sheet to load.
    :return: Tuple of DataFrames (with and without headers).
    """
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    data2 = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    return data, data2

def determine_finishing_row(data, file_path, sheet_name):
    """
    We want to find the finishing row of the tables (we don't know sine it is an excel file without headers)
    We want to consider rows that have at least one non-NaN value:
    param data: DataFrame to analyze.
    :return: finishing row index
    """
    
    #There is an exception for file_path './Design_C_ATS.xlsx'. There are additional informations only in this file that is not needed.
    if(file_path == './Files/Raw_Files/Design_C_ATS.xlsx' and sheet_name == 0):
        return 284
        
    return data.dropna(how='all').index[-1] + 1

def prepare_data_to_csv1(file_path, sheet_name, data_D):
    """
    Prepare data for CSV export (C part 1).
    This function create the csv file with Information A and B as X and some part of information C

    :param file_path: Path to the source Excel file.
    :param sheet_name: Sheet name to process.
    :param data_D: DataFrame containing additional data needed for processing.
    :return: DataFrame ready for CSV export.
    """
    
    data, data2 = load_excel_data(file_path, sheet_name)
    nbr_building = compute_nbr_building(data)
    df = parse_header_data(data, nbr_building)
    finishing_row_informationB = determine_finishing_row(data, file_path, sheet_name)
    filling_values(data2, starting_row_informationB, finishing_row_informationB)
    nbr_walls = finishing_row_informationB - starting_row_informationB

    #Add type A information
    repeated_df = pd.DataFrame(np.repeat(df.values, nbr_walls, axis=0), columns=df.columns)

    #Add type B information
    df1 = [data2.iloc[starting_row_informationB:finishing_row_informationB, 3:9]
           .rename(columns={data2.columns[3]: "Story",
                            data2.columns[4]: "Direction",
                            data2.columns[5]: "Wall",
                            data2.columns[6]: "L cm",
                            data2.columns[7]: "xi cm",
                            data2.columns[8]: "yi cm"}) for _ in range(nbr_building)]

    #Add type C information
    dfs = [data2.iloc[starting_row_informationB:finishing_row_informationB,
                      9 + size_columns_informationA * i: 14 + size_columns_informationA * i]
           .rename(columns={data2.columns[9 + size_columns_informationA * i]: "Nail spacing [cm]",
                            data2.columns[10 + size_columns_informationA * i]: "Number sheathing panels",
                            data2.columns[11 + size_columns_informationA * i]: "Number end studs",
                            data2.columns[12 + size_columns_informationA * i]: "Total number studs",
                            data2.columns[13 + size_columns_informationA * i]: "HoldDown Model / ATS"}) for i in range(nbr_building)]

    result2 = pd.concat(df1, ignore_index=True)
    result3 = pd.concat(dfs, ignore_index=True)

    d_plus_quarter_l_values = []
    story_area_values = []

    for i in range(0, nbr_building):
        for j in range(0, finishing_row_informationB - starting_row_informationB):
            story = int(result2.iat[j, 0])
            d_plus_quarter_l = data2.iat[4 + story, 11 + size_columns_informationA * i]
            d_plus_quarter_l_values.append(d_plus_quarter_l)
            story_area = data2.iat[4 + story, 13 + size_columns_informationA * i]
            story_area_values.append(story_area)

    result2['D+0.25L'] = d_plus_quarter_l_values
    result2['Story Area'] = story_area_values

    repeated_df = repeated_df.drop('header', axis=1)
    resultFinal = pd.concat([repeated_df, result2, result3], axis=1, ignore_index=False)

    return resultFinal


def find_DL_Story_Txy_walls(df, d_all, columns_all, d_walls,data2,nbr_building):
    """
    Compiles and merges various data sources into a single DataFrame.

    This function processes and combines data related to building features (like wall dimensions and story areas),
    along with Tx and Ty values, to create a comprehensive DataFrame representing all the information.

    :param df: DataFrame containing the initial building data, excluding headers.
    :param d_all: List of lists containing wall dimensions and other related data for each building.
    :param columns_all: List of column names corresponding to the data in d_all.
    :param d_walls: List containing the number of walls for each building.
    :param data2 : 
    :param nbr_building : Number of buildings (headers) to process.
    :return: A DataFrame that combines the input data into a structured format.
    """
    df_all = pd.DataFrame(d_all, columns=columns_all)
    # Assuming data2, nbr_building, and size_columns_informationA are globally defined
    unique_values = data2.iloc[:, 3].unique()[2:]
    d_plus_quarter_l_values = np.zeros((nbr_building, len(unique_values)))
    story_area_values = np.zeros((nbr_building, len(unique_values)))
    Tx_values = []
    Ty_values = []

    for i in range(0, nbr_building):
        for j, value in enumerate(unique_values):
            story = int(value)
            d_plus_quarter_l = data2.iat[4 + story, 11 + size_columns_informationA * i]
            d_plus_quarter_l_values[i, j] = d_plus_quarter_l
            story_area = data2.iat[4 + story, 13 + size_columns_informationA * i]
            story_area_values[i, j] = story_area

        Tx_values.append(data2.iat[row_Tx_Ty_values, 9 + size_columns_informationA * i])
        Ty_values.append(data2.iat[row_Tx_Ty_values, 10 + size_columns_informationA * i])

    df = df.drop('header', axis=1)
    resultFinal = pd.concat([df, df_all], axis=1, ignore_index=False)
    df_d_plus_quarter_l = pd.DataFrame(d_plus_quarter_l_values)
    df_d_plus_quarter_l.columns = [f'D+0.25L {i + 1}' for i in range(len(unique_values))]
    story_area_values = pd.DataFrame(story_area_values)
    story_area_values.columns = [f'Story Area {i + 1}' for i in range(len(unique_values))]
    df_nbr_walls, Tx_values, Ty_values = pd.DataFrame(d_walls), pd.DataFrame(Tx_values), pd.DataFrame(Ty_values)
    df_nbr_walls.columns, Tx_values.columns, Ty_values.columns = ['Total_Number_walls'], ['Tx(s)'], ['Ty(s)']

    return pd.concat([resultFinal, df_d_plus_quarter_l, story_area_values, df_nbr_walls, Tx_values, Ty_values], axis=1, ignore_index=False)


def prepare_data_from_excel(file_path, sheet_name, performance_data):
    """
    Prepare data from an Excel file (C part 2).
    This function create the csv file with Information A and B as X and the rest of information C as Y, especially only Tx and Ty
    Function to prepare data from an Excel file and return a DataFrame

    :param file_path: Path to the source Excel file.
    :param sheet_name: Sheet name to process.
    :param performance_data: DataFrame containing performance data.
    :return: DataFrame ready for CSV export.
    """
    data, data2 = load_excel_data(file_path, sheet_name)
    nbr_building = compute_nbr_building(data)
    df = parse_header_data(data, nbr_building)
    finishing_row_informationB = determine_finishing_row(data, file_path, sheet_name)
    filling_values(data2, starting_row_informationB, finishing_row_informationB)
    nbr_walls = finishing_row_informationB - starting_row_informationB
    
    d_all = []
    columns_all = []
    d_walls = []

    for i in range(nbr_building):
        d_all_bis = []

        for j in range(starting_row_informationB, finishing_row_informationB + 1):
            d_all_bis += [data2.iat[j, 6], data2.iat[j, 7], data2.iat[j, 8]]

            if i == 0:
                name_plus = '_' + str(data2.iat[j, 3]) + '_' + str(data2.iat[j, 4]) + '_' + str(data2.iat[j, 5])
                columns_all += [
                    'L cm' + name_plus,
                    'xi cm' + name_plus,
                    'yi cm' + name_plus
                ]

        d_all.append(d_all_bis)
        d_walls += [nbr_walls]

    resultFinal = find_DL_Story_Txy_walls(df, d_all, columns_all, d_walls,data2,nbr_building)
    return resultFinal

def prepare_data_to_csv3(file_path, sheet_name, data_D):
    """
    Prepare data for CSV export (D part).

    :param file_path: Path to the source Excel file.
    :param sheet_name: Sheet name to process.
    :param data_D: DataFrame containing additional data needed for processing.
    :return: DataFrame ready for CSV export.
    """
    data, data2 = load_excel_data(file_path, sheet_name)
    nbr_building = compute_nbr_building(data)
    df = parse_header_data(data, nbr_building)
    finishing_row_informationB = determine_finishing_row(data, file_path, sheet_name)
    filling_values(data2, starting_row_informationB, finishing_row_informationB)
    repetitions = finishing_row_informationB - starting_row_informationB
    d_all = []
    columns_all = []
    d_walls = []
    for i in range(nbr_building) :
        d_all_bis = []
        for j in range(starting_row_informationB, finishing_row_informationB+1) :
            d_all_bis += [data2.iat[j,6], data2.iat[j,7], data2.iat[j,8], 
                          data2.iat[j,9 + size_columns_informationA * i],data2.iat[j,10 + size_columns_informationA * i],
                          data2.iat[j,11 + size_columns_informationA * i],data2.iat[j,12 + size_columns_informationA * i],
                          data2.iat[j,13 + size_columns_informationA * i]
                         ]
            if i==0 :
                name_plus = '_' + str(data2.iat[j,3])+ '_' + str(data2.iat[j,4]) +'_'+ str(data2.iat[j,5])
                columns_all += ['L cm' + name_plus,
                                'xi cm'+ name_plus,
                                'yi cm'+ name_plus,
                                "Nail spacing [cm]" + name_plus ,
                                "Number sheathing panels"+ name_plus,
                                "Number end studs"+name_plus,
                                "Total number studs"+name_plus,
                                "HoldDown Model / ATS "+name_plus
                               ]

        d_all.append(d_all_bis)
        d_walls += [repetitions]
 
    #Add type D information (for the second prediction)
    all_rows_data = []
    for index, row in df.iterrows():
        # Taking the header in each row
        header_value = row['header']
        
        # Find corresponding data in data_D using the header value
        row_header_D = find_performance_using_header(data_D, header_value)
        
        # Append the found data to the list
        all_rows_data.append(row_header_D)
    
    df_D = pd.DataFrame(all_rows_data, columns=columns_D)
   

    resultFinal = resultFinal = find_DL_Story_Txy_walls(df, d_all, columns_all, d_walls,data2,nbr_building)
    resultFinal = pd.concat([resultFinal, df_D], axis=1, ignore_index=False)
    return resultFinal

def process_files_and_merge(file_info, process_function, data_D, merge_function):
    """
    Process files, merge the results using a specified merge function.

    :param file_info: List of tuples with file paths and sheet indices.
    :param process_function: Function to process each file.
    :param data_D: DataFrame containing additional data needed for processing.
    :param merge_function: Function to merge processed data frames.
    :return: Merged DataFrame.
    """
    processed_files = process_files(file_info, process_function, data_D)
    return merge_function(processed_files)

if __name__ == "__main__":

    path = './Files/Raw_Files'
    files = [
            '/Design_P_ATS.xlsx', '/Design_P_HD.xlsx',
            '/Design_D_ATS.xlsx', '/Design_D_HD.xlsx',
            '/Design_C_ATS.xlsx', '/Design_C_HD.xlsx',
            '/Design_Q_ATS.xlsx', '/Design_Q_HD.xlsx'
    ]

    data_D = pd.read_excel(path + '/PerformanceResults.xlsx', header=None)
    file_info = get_file_information(files, path)

    output_path = "Files/Before_Feature_Engineering"


    #If the output directory does not exist , we create it 
    if not os.path.exists(output_path):
        print("Creating the folder :"+output_path)
        os.makedirs(output_path)


        

    # The folder (Files/Before_Feature_Engineering) must exists beforehands

    # Process, merge using merge_dataframes1, and save C part 1 data
    merged_df_C_part1 = process_files_and_merge(file_info, prepare_data_to_csv1, data_D, merge_dataframes1)
    save_to_csv(merged_df_C_part1, output_path+'/data_C_part1.csv')

    # Process, merge using merge_dataframes1, and save C part 2 data
    merged_df_C_part2 = process_files_and_merge(file_info, prepare_data_from_excel, data_D, merge_dataframes1)
    # Optionally reorder columns for merged_df_C_part2 if necessary

    save_to_csv(merged_df_C_part2, output_path+'/data_C_part2.csv')

    # Process, merge using merge_dataframes2, and save D data
    merged_df_D = process_files_and_merge(file_info, prepare_data_to_csv3, data_D, merge_dataframes2)
    # Optionally reorder columns for merged_df_D if necessary

    save_to_csv(merged_df_D, output_path+'/data_D.csv')