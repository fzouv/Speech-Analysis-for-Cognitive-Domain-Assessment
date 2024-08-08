import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

file_path = '/Users/fotianazouvani/Downloads/SLP/Diss/complete_table.csv'
data = pd.read_csv(file_path)

#preprocessing data
data = data.dropna()
data = data.drop(columns=['MoCA Total'])
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

# predictive power data
predictive_power_path = '/Users/fotianazouvani/PycharmProjects/pythonProject12/predictive_power.csv'
predictive_power = pd.read_csv(predictive_power_path, index_col=0)


# moca tasks
tasks = [
    'Trail making task', 'Copy cube task', 'Draw clock task', 'Naming task',
    'Digits task', 'Letter attention task', 'Serial 7 task', 'Repetition task',
    'Verbal fluency scoring task', 'Abstraction task', 'Delayed recall task', 'Orientation task'
]

# dataframe to store predictions vs true values
predictions = data[['Record ID']].copy()
results = []


#training and evaluating model
def train_and_evaluate_model(X, y, task, top_features=None):
    # select only top features
    if top_features is not None:
        X = X[top_features]

    # feature standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train-test sets split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # rf regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #model evaluation
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    return model, mae, mse, r2


#evaluation pre fine-tune
original_results = []
for task in tasks:
    y = data[task]
    X = data.drop(columns=tasks + ['Record ID'])

    model, mae, mse, r2 = train_and_evaluate_model(X, y, task)
    original_results.append({'Task': task, 'MAE': mae, 'MSE': mse, 'R^2': r2})
    joblib.dump(model, f'moca_model_{task.replace(" ", "_")}_original.pkl')

#evaluation post fine-tune
fine_tuned_results = []
for task in tasks:
    y = data[task]
    top_features = predictive_power[task].sort_values(ascending=False)[:20].index.tolist()
    X = data[top_features]

    model, mae, mse, r2 = train_and_evaluate_model(X, y, task, top_features)
    fine_tuned_results.append({'Task': task, 'MAE': mae, 'MSE': mse, 'R^2': r2})
    joblib.dump(model, f'moca_model_{task.replace(" ", "_")}_top20.pkl')


original_results_df = pd.DataFrame(original_results)
fine_tuned_results_df = pd.DataFrame(fine_tuned_results)

# store results to csvs
original_results_df.to_csv('original_model_results.csv', index=False)
fine_tuned_results_df.to_csv('fine_tuned_model_results.csv', index=False)


comparison_df = original_results_df.copy()
comparison_df.columns = ['Task', 'Original_MAE', 'Original_MSE', 'Original_R^2']
comparison_df = comparison_df.merge(fine_tuned_results_df, on='Task')
comparison_df.columns = ['Task', 'Original_MAE', 'Original_MSE', 'Original_R^2', 'Fine_Tuned_MAE', 'Fine_Tuned_MSE',
                         'Fine_Tuned_R^2']

print(comparison_df)

comparison_df.to_csv('model_comparison_results.csv', index=False)