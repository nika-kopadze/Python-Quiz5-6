import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('jobs_in_data.csv')

############# დავალება 1 #################################################:
X_simple = df[['experience_level']]  
y_simple = df['salary_in_usd']

X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train_simple)

y_pred_simple = simple_model.predict(X_test_simple)

mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)

print("Task 1 - Simple Linear Regression:")
print("Mean Squared Error:", mse_simple)
print("R-squared:", r2_simple)

############# დავალება 2 #################################################:
X_multi = df[['experience_level', 'work_year', 'company_size']]
y_multi = df['salary_in_usd']

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

multi_model = LinearRegression()
multi_model.fit(X_train_multi, y_train_multi)

y_pred_multi = multi_model.predict(X_test_multi)

mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)

print("\nTask 2 - Multiple Linear Regression:")
print("Mean Squared Error:", mse_multi)
print("R-squared:", r2_multi)

############# დავალება 3 #################################################:
X_tree = df[['experience_level', 'work_year', 'company_size']]
y_tree = df['salary_in_usd']

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.2, random_state=42)

tree_model = DecisionTreeRegressor()
tree_model.fit(X_train_tree, y_train_tree)

y_pred_tree = tree_model.predict(X_test_tree)

mse_tree = mean_squared_error(y_test_tree, y_pred_tree)
r2_tree = r2_score(y_test_tree, y_pred_tree)

print("\nTask 3 - Decision Tree Regression:")
print("Mean Squared Error:", mse_tree)
print("R-squared:", r2_tree)

############# დავალება 4 #################################################:
X_logistic = df[['experience_level', 'work_year', 'company_size']]
y_logistic = df['job_category']

X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X_logistic, y_logistic, test_size=0.2, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X_train_logistic, y_train_logistic)

y_pred_logistic = logistic_model.predict(X_test_logistic)

accuracy_logistic = accuracy_score(y_test_logistic, y_pred_logistic)

print("\nTask 4 - Logistic Regression:")
print("Accuracy:", accuracy_logistic)

############# დავალება 5 #################################################:
X_tree_class = df[['experience_level', 'work_year', 'company_size']]
y_tree_class = df['job_category']

X_train_tree_class, X_test_tree_class, y_train_tree_class, y_test_tree_class = train_test_split(X_tree_class, y_tree_class, test_size=0.2, random_state=42)

tree_class_model = DecisionTreeClassifier()
tree_class_model.fit(X_train_tree_class, y_train_tree_class)

y_pred_tree_class = tree_class_model.predict(X_test_tree_class)

accuracy_tree_class = accuracy_score(y_test_tree_class, y_pred_tree_class)

print("\nTask 5 - Decision Tree Classification:")
print("Accuracy:", accuracy_tree_class)