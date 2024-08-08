import pandas as pd
import numpy as np
from scipy.stats import pearsonr


file_path = '/Users/fotianazouvani/Downloads/SLP/Diss/complete_table.csv'  # Update this path
data = pd.read_csv(file_path)

#preprocessing (removing demographic info and other irrelevant info)
irrelevant_columns = [
    'Record ID',
    'Age',
    'Sex',
    'Add point if 12 years or less of education.',
    'MoCA Total'
]

# moca tasks
moca_tasks = ['Trail making task', 'Copy cube task', 'Draw clock task', 'Naming task',
              'Digits task', 'Letter attention task', 'Serial 7 task', 'Repetition task',
              'Verbal fluency scoring task', 'Abstraction task', 'Delayed recall task', 'Orientation task']

#preprocessing data (dropping irrelevant columns and removing missing values)
data_cleaned = data.drop(columns=irrelevant_columns)
data_cleaned = data_cleaned.replace([np.inf, -np.inf], np.nan).dropna()

#speech features
speech_features = [col for col in data_cleaned.columns if col not in moca_tasks and col != 'Record ID']

# dataframe for correlation coefficients
correlation_results = pd.DataFrame(index=speech_features, columns=moca_tasks)

#computing Pearson's correlation for each feature-task pair
for task in moca_tasks:
    for feature in speech_features:
        if data_cleaned[feature].nunique() > 1:  # checking if column is not constant
            corr, _ = pearsonr(data_cleaned[feature], data_cleaned[task])
            correlation_results.at[feature, task] = corr
        else:
            correlation_results.at[feature, task] = None  #if the column is constant set to none

correlation_results = correlation_results.astype(float).round(4)
print(correlation_results)

# saving into table
correlation_results.to_csv('/Users/fotianazouvani/Downloads/SLP/Diss/correlation_results.csv', index=True)
