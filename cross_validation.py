import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

file_path = '/Users/fotianazouvani/Downloads/SLP/Diss/complete_table.csv'
data = pd.read_csv(file_path)

# data preprocessing
data = data.dropna()
data = data.drop(columns=['MoCA Total'])
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

# data showing predictive power of features
predictive_power_path = '/Users/fotianazouvani/PycharmProjects/pythonProject12/predictive_power.csv'
predictive_power = pd.read_csv(predictive_power_path, index_col=0)

# all moca tasks
tasks = [
    'Trail making task', 'Copy cube task', 'Draw clock task', 'Naming task',
    'Digits task', 'Letter attention task', 'Serial 7 task', 'Repetition task',
    'Verbal fluency scoring task', 'Abstraction task', 'Delayed recall task', 'Orientation task'
]


# 5-fold cross-validation
def perform_cross_validation(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    neg_mae_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    neg_mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    return -np.mean(neg_mae_scores), -np.mean(neg_mse_scores), np.mean(r2_scores)


# evaluating before and after fine-tuning
original_results = []
fine_tuned_results = []

for task in tasks:
    y = data[task]
    X_original = data.drop(columns=tasks + ['Record ID'])
    top_features = predictive_power[task].sort_values(ascending=False)[:20].index.tolist()
    X_fine_tuned = data[top_features]

    # feature standardisation
    scaler_original = StandardScaler()
    X_original_scaled = scaler_original.fit_transform(X_original)

    scaler_fine_tuned = StandardScaler()
    X_fine_tuned_scaled = scaler_fine_tuned.fit_transform(X_fine_tuned)

    # cross-val for original modle
    original_mae, original_mse, original_r2 = perform_cross_validation(X_original_scaled, y)
    original_results.append({'Task': task, 'MAE': original_mae, 'MSE': original_mse, 'R^2': original_r2})

    # cross-val for fine-tuned model
    fine_tuned_mae, fine_tuned_mse, fine_tuned_r2 = perform_cross_validation(X_fine_tuned_scaled, y)
    fine_tuned_results.append({'Task': task, 'MAE': fine_tuned_mae, 'MSE': fine_tuned_mse, 'R^2': fine_tuned_r2})

original_results_df = pd.DataFrame(original_results)
fine_tuned_results_df = pd.DataFrame(fine_tuned_results)

# saving results as csv
original_results_df.to_csv('/Users/fotianazouvani/Downloads/SLP/Diss/original_model_cv_results.csv', index=False)
fine_tuned_results_df.to_csv('/Users/fotianazouvani/Downloads/SLP/Diss/fine_tuned_model_cv_results.csv', index=False)

# comparing
comparison_df = original_results_df.copy()
comparison_df.columns = ['Task', 'Original_MAE', 'Original_MSE', 'Original_R^2']
comparison_df = comparison_df.merge(fine_tuned_results_df, on='Task')
comparison_df.columns = ['Task', 'Original_MAE', 'Original_MSE', 'Original_R^2', 'Fine_Tuned_MAE', 'Fine_Tuned_MSE',
                         'Fine_Tuned_R^2']

# save comparison as csv file
comparison_df.to_csv('/Users/fotianazouvani/Downloads/SLP/Diss/model_cv_comparison_results.csv', index=False)

# performance improvements
comparison_df['MAE_Improvement'] = comparison_df['Original_MAE'] - comparison_df['Fine_Tuned_MAE']
comparison_df['MSE_Improvement'] = comparison_df['Original_MSE'] - comparison_df['Fine_Tuned_MSE']
comparison_df['R2_Improvement'] = comparison_df['Fine_Tuned_R^2'] - comparison_df['Original_R^2']

# visualising performance improvements
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# MAE improvement
axes[0].barh(comparison_df['Task'], comparison_df['MAE_Improvement'], color='skyblue')
axes[0].set_title('MAE Improvement (Original - Fine-Tuned)')
axes[0].set_xlabel('Improvement in MAE')
axes[0].invert_yaxis()

# MSE improvement
axes[1].barh(comparison_df['Task'], comparison_df['MSE_Improvement'], color='salmon')
axes[1].set_title('MSE Improvement (Original - Fine-Tuned)')
axes[1].set_xlabel('Improvement in MSE')
axes[1].invert_yaxis()

# R2 improvement
axes[2].barh(comparison_df['Task'], comparison_df['R2_Improvement'], color='lightgreen')
axes[2].set_title('R^2 Improvement (Fine-Tuned - Original)')
axes[2].set_xlabel('Improvement in R^2')
axes[2].invert_yaxis()

plt.tight_layout()
plt.show()

print(comparison_df)