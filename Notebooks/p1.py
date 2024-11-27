'''
First step for preprocessig the data from the ETDD70 dataset (https://zenodo.org/records/13332134).
This script will merge all the files with the suffix '_fixations.csv' in the folder specified by the user.
'''

import os
import sys
import pandas as pd

import os
import sys
import pandas as pd

def merge_csv_files(folder_path, output_file):
    # List to hold dataframes
    dataframes = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a CSV file and contains '_fixations'
        if filename.endswith(".csv") and "_fixations" in filename:
            file_path = os.path.join(folder_path, filename)
            print(f"Reading file: {filename}")
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Concatenate all dataframes
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        # Write the merged dataframe to a new CSV file
        merged_df.to_csv(output_file, index=False)
        print(f"Successfully merged files into {output_file}")
    else:
        print("No files with '_fixations.csv' found.")

def main(**kwargs):
    folder_path = kwargs.get('folder_path')
    output_file = folder_path + "/merged_fixations.csv"
    merge_csv_files(folder_path, output_file)

if __name__ == '__main__':
    folder_path = sys.argv[1]
    main(folder_path=folder_path)