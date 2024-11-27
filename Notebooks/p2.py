'''
Second step for preprocessig the data from the ETDD70 dataset (https://zenodo.org/records/13332134).
This script will merge the 'merged_fixations.csv' file (result of p1.py) with the 'dyslexia_class_label.csv' file.
'''

import pandas as pd

# Load the CSV files
filepath = 'C:/Users/nickp/Documents/Courses/P1&P2/EyeMovementDat/data/merged_fixations.csv'
file1 = pd.read_csv(filepath)
file2 = pd.read_csv('C:/Users/nickp/Documents/Courses/P1&P2/EyeMovementDat/data/dyslexia_class_label.csv')

# Merge the two dataframes on 'sid' and 'subject_id' 
merged_df = pd.merge(file1, file2, left_on='sid', right_on='subject_id', how='left')

# Drop the 'subject_id' column from the merged dataframe
merged_df = merged_df.drop(columns=['subject_id'])

# Save the merged result
merged_df.to_csv(filepath, index=False)

print("Merging complete!")
