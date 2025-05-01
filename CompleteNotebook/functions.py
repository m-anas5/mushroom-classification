## Function File

## essential Libraries for Data manipulation and handling
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from tabulate import tabulate
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

## function for reading a csv and returning data frame
def read_csv(file_path):
    df = pd.read_csv("mushrooms.csv")
    return df

## function for storing head, describe, info in seperate text files so they can be retreived and printed without missing columns
def store_information(df):
    with open("dfhead.txt", "w") as f:
        f.write("-" * 30 + "Dataframe Head" + "-" * 30 + "\n\n")
        f.write(df.head().to_string())
    with open("dfdescribe.txt", "w") as f:
        f.write("-" * 30 + "Dataframe Description" + "-" * 30 + "\n\n")
        f.write(df.describe().to_string())
    with open("dfinformatin.txt", "w") as f:
        f.write("-" * 30 + "Dataframe Information" + "-" * 30 + "\n\n")
        df.info(buf=f)

## function to print contents of text file.
def print_file_content(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            print(content)
        print("_" * 80)
    except FileNotFoundError:
        print(f"File not found {file_path}")

## function to get the plots for the object columns
def get_plots(df):
    
    object_columns = df.select_dtypes(include='object').columns
    n_cols = 5
    n_rows = math.ceil(len(object_columns) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 9 * n_rows))
    axes = axes.flatten()  

    for ax, col in zip(axes, object_columns):
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.tick_params(axis='x', rotation=45)

    for i in range(len(object_columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

## Printing all the unique values present in each column
def printing_column(df):
    for col in df.columns:
        print(f"{col} unique values:\n{df[col].unique()}\n")

def show_invalid_entries(df, column, invalid_list, after=False):
    invalid_entries = df[df[column].isin(invalid_list)]
    if after:
        print(f"After cleaning — number of invalid {column} entries: {len(invalid_entries)}")
    else:
        print(f"Number of invalid {column} entries: {len(invalid_entries)}")
    print(invalid_entries[column].value_counts())

## Function to display plots of target column vs all other features
def targetVfeatures(target_col, df):
    # Get all feature columns except the target
    feature_cols = [col for col in df.columns if col != target_col]

    # Number of plots per row
    plots_per_row = 5
    total_plots = len(feature_cols)
    rows = math.ceil(total_plots / plots_per_row)

    # Set figure size based on number of rows and columns
    plt.figure(figsize=(plots_per_row * 4, rows * 4))

    for i, col in enumerate(feature_cols):
        # Group and unstack to prepare for bar plot
        counts = df.groupby([col, target_col]).size().unstack(fill_value=0)

        # Plot
        plt.subplot(rows, plots_per_row, i + 1)
        counts.plot(kind='bar', ax=plt.gca(), title=col)
        plt.xlabel(col)
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

## function to represent target column against features to represent as percentages.
def target_percentage_tables(target_col, df):
    result = {}

    # Get all feature columns except the target
    feature_cols = [col for col in df.columns if col != target_col]

    for col in feature_cols:
        # Get counts and compute percentages
        counts = df.groupby([col, target_col]).size().unstack(fill_value=0)
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        result[col] = percentages.round(2)  

    return result

## printing in a tabular form using tabulate library.
def print_target_distributions(target_col, df):
    tables = target_percentage_tables(target_col, df)

    for feature, table in tables.items():
        print(f"\nFeature: {feature}")
        print(tabulate(table, headers='keys', tablefmt='fancy_grid'))

## function that takes in model, X = features, y = target and prints the accuracy and classification report for that model
def classifyModel(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

## function for cross validation, it takes in model, X = features, y = target and prints the accuracy and classification report for that model
## the dataset is split into 4 parts. It also produces mean accuracy after cross validation
def cross_validate_decision_tree(model, X, y, n_splits=4):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) 
    reports = []
    accuracies = []
    all_reports = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}:")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print("Accuracy:", accuracy)

        table_data = []
        for class_label, metrics in report.items():
            if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                table_data.append([class_label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])

        header = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        print("Classification Report:")
        print(tabulate(table_data, headers=header, floatfmt=".4f"))
        print("-" * 40)

        all_reports.append(report)
        accuracies.append(accuracy)

    print("Mean Performance:")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")

