#training the model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

def train():
    # Load the dataset
    train = pd.read_csv("db/names.csv")
    print("Original data shape:", train.shape)
    
    # Step 1: Check for missing values and handle them
    if train.isnull().sum().any():
        print("Missing values detected. Filling with 0.")
        train.fillna(0, inplace=True)
    
    # Step 2: Save specified columns before one-hot encoding
    columns_to_preserve = ['name1', 'name2']
    preserved_columns = train[columns_to_preserve]
    
    # Remove the specified columns from the DataFrame
    train = train.drop(columns=columns_to_preserve)
    
    # Separate labels from features
    labels = train.pop("label")
    
    # Check class distribution
    class_distribution = labels.value_counts()
    print("Class Distribution:\n", class_distribution)
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=0.25, random_state=34, stratify=labels)
    
    # Define parameter grid for Grid Search
    param_grid = {
        'n_estimators': [145,146],  # Increase the number of trees
        'max_depth': [22,21],  # Experiment with different depths
        'min_samples_split':[20,10],
        'min_samples_leaf': [16,21],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'oob_score': [True, False],
        'n_jobs': [-1],
        'warm_start': [True, False]
    
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
    
    # Fit the model with grid search
    grid_search.fit(x_train, y_train)
    
    # Retrieve the best estimator
    best_rf = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
    
    # Fit the model with grid search
    grid_search.fit(x_train, y_train)
    
    # Retrieve the best estimator
    best_rf = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    
    # Make predictions with the best model
    y_pred = best_rf.predict(x_test)
    
    # Calculate and print the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the best model: {accuracy:.2f}")
    
    # Print classification report for detailed analysis
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Display confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", confusion)
    
    # Combine y_test and y_pred with associated names for better understanding
    test_results = preserved_columns.iloc[y_test.index].copy()
    test_results['y_test'] = y_test.values
    test_results['y_pred'] = y_pred
    
    # Print y_test and y_pred values with associated names
    print("\nNames with y_test and y_pred values:")
    print(test_results[['name1', 'name2', 'y_test', 'y_pred']])
    
    # Filter out mismatched results where y_pred does not match y_test
    mismatched_results = test_results[test_results['y_test'] != test_results['y_pred']]
    
    # Print only the mismatched results
    print("\nMismatched Names with y_test and y_pred values:")
    print(mismatched_results[['name1', 'name2', 'y_test', 'y_pred']])
    
    # Save the mismatched results as a new DataFrame (optional)
    mismatched_results.to_csv("mismatched_results.csv", index=False)
    
    # Analyze feature importance
    feature_importances = best_rf.feature_importances_
    feature_names = train.columns
    
    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.show()
    
    # Save the trained model to a .pkl file
    joblib.dump(best_rf, 'db/best_random_forest_model.pkl')
    print("Best model saved as 'best_random_forest_model.pkl'.")
    