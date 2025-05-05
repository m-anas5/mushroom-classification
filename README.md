## Mushroom Classification Using Machine Learning: Edible vs Poisnous

This project applies machine learning algorithms to classify mushrooms as edible or poisnous based on their physical characteristics. This dataset was taken from Kaggle [link](https://www.kaggle.com/datasets/uciml/mushroom-classification)


## Overview
This project focuses on the Mushroom Dataset from the UCI Machine Learning Repository, which contains 22 categorical attributes describing mushrooms and classifies them as either edible or poisonous. The problem is approached as a binary classification task, using Logistic Regression as a baseline and Decision Tree as the primary model. Both models achieved 100% accuracy, indicating that the dataset is relatively simple and the features provide strong signals for classification. Although the results are not surprising, the project demonstrates how basic machine learning models can perform exceptionally well on clean, well-structured data.


## Summary of Workdone

### Data

* Data:
  * Type: CSV file. 
    * Input: CSV file containing 22 categorical features that describe physical characteristics of mushrooms.
    * Output: A label indicating whether each mushroom is edible (e) or poisonous (p).
  * Size: 
    * The dataset contains a total of 8,124.
  * Instances (Train, Test, Validation Split): The data was split into training (80%) and testing (20%), resulting in approximately 6,499 training samples and 1,   625 test samples. No separate validation set was used; however, 4-fold cross-validation was applied during model evaluation to improve reliability and reduce overfitting.


#### Preprocessing / Clean up

As part of data preparation, two columns were removed from the dataset: veil-type and stalk-root. The veil-type column was dropped because it contained only a single unique value across all instances, making it non-informative for classification. The stalk-root column was removed due to the presence of 2,480 missing values, which constituted a significant portion of the dataset. Removing these columns helped simplify the feature space and avoid the introduction of bias or noise from imputation.

#### Data Visualization
![Plot of target column against otehr features](/2.png)
Graph 2.1 Plots of target column against all other features, showing the distribution.

### Problem Formulation

The mushroom classification task was formulated as a supervised binary classification problem, where the input is a set of 22 categorical features describing mushroom characteristics, and the output is a binary label indicating whether the mushroom is edible (e) or poisonous (p). The dataset was preprocessed to remove two non-informative or incomplete columns, and the remaining categorical features were encoded using one hot encoding to convert them into a numerical format suitable for machine learning models.

Two models were implemented and compared: Logistic Regression and a Decision Tree Classifier. Logistic Regression was chosen for its simplicity and interpretability as a baseline model. The Decision Tree was selected due to its natural handling of categorical data and ability to model non-linear relationships. For further evaluation, 4-fold cross-validation was applied to the Decision Tree to assess its generalization. Additionally, hyperparameter tuning (e.g., setting criterion='entropy' and max_depth=3) was explored to evaluate how limiting complexity affects performance. Since both models achieved high accuracy, no custom loss functions or optimizers were required; standard accuracy and classification reports were used as evaluation metrics.

### Training

Training was performed using Python with scikit-learn on a standard CPU-based environment. Since the dataset is relatively small and both models are computationally efficient, training time was negligible each model trained in under a second. Due to the deterministic and non-iterative nature of Decision Trees and Logistic Regression in scikit-learn, there were no training epochs or learning curves to monitor as would be the case in deep learning. Therefore, model evaluation was conducted through accuracy scores and classification metrics rather than epoch-based monitoring.

There were no major difficulties during training, but handling the categorical features required attention. The dataset’s clean structure (no missing values after preprocessing and well-separated classes) contributed to the models achieving nearly perfect accuracy, making this an ideal example for demonstrating classification techniques on categorical data.

### Performance Comparison
Logistic Regression:

Accuracy: 1.0

Classification Report:

              precision    recall  f1-score   support

           e       1.00      1.00      1.00       843
           p       1.00      1.00      1.00       782

    accuracy                           1.00      1625  
    macro avg      1.00      1.00      1.00      1625
    weighted avg   1.00      1.00      1.00      1625

Decision Tree Classifier:

Accuracy: 1.0

Classification Report:

              precision    recall  f1-score   support

           e       1.00      1.00      1.00       843
           p       1.00      1.00      1.00       782

    accuracy                           1.00      1625
    macro avg      1.00      1.00      1.00      1625
    weighted avg   1.00      1.00      1.00      1625

Decision Tree Classifier with max_depth = 3:

Accuracy: 0.96
Classification Report:

              precision    recall  f1-score   support

           e       0.94      1.00      0.96       843
           p       1.00      0.93      0.96       782

    accuracy                           0.96      1625
    macro avg      0.97      0.96      0.96      1625
    weighted avg   0.96      0.96      0.96      1625

### Conclusions

The mushroom dataset proved to be well-structured and required minimal preprocessing. Aside from one column with missing values, the data was clean and ready for modeling. A Decision Tree Classifier was employed, and its performance was evaluated using 4-fold cross-validation. Additionally, hyperparameter tuning was performed to assess whether model optimization could improve results. However, both cross-validation and tuning had little effect on accuracy or other evaluation metrics. This indicates that the dataset has highly separable classes, allowing even simple models like Logistic Regression and Decision Trees to achieve near-perfect classification performance.

### Future Work:

**Imputation Strategies:**

Instead of dropping the stalk-root column due to missing values, explore imputation methods (e.g., using most frequent value, or predictive imputation) to see if including it improves or changes model behavior.

**Get new test data:**
Get new test data and validate on existing models.

## How to reproduce results
To reproduce the results in this project, run the Jupyter notebook Document.ipynb located in the CompleteNotebook folder. This notebook includes all steps from data loading and preprocessing to model training, evaluation, and cross-validation. All code is executed from scratch—no pre-trained models or cached outputs are used.

This project was developed and tested on a local machine using Jupyter Notebook within Visual Studio Code. The main Python libraries used include: pandas, numpy, seaborn, matplotlib, scikit-learn, and tabulate. Make sure these dependencies are installed before running the notebook.

The accompanying functions.py file contains reusable helper functions and should be kept in the same folder as the notebook for proper execution.

This repository is released under the MIT License, which allows others to freely use, modify, and distribute the code. Users are encouraged to adapt the pipeline to their own datasets or experiments if desired.

### Overview of files in repository

The repository consist of two folders. CompleteNotebook and ScratchNotebook, CompleteNotebook consist of:
**CompleteNotebook:**
Total files: 7 files

- Document.ipynb, this notebook contains all essential implementations to evaluate and do calculations.
- functions.py, this python file contain all essential functions needed to shrink the code in the notebook.
- mushrooms.csv, csv file used.
- 4 text files: df_head_output.txt, dfdescribe.txt, dfhead.txt, dfinformation.txt. As the names suggest these are output files just to visualize the dataset.

**ScratchNotebook:**
Total files:  4 files
- Project_P2_pt1.ipynb, contains inital visualization
- Project_P2_pt2.ipynb, walk through preprocessing
- Project_P2_pt3.ipynb, displays different models and their performance
- mushrooms.csv

**Note:** The ScratchNotebook folder is for exploratory purposes only. For the complete and cleaned implementation, refer to the CompleteNotebook folder and the Document.ipynb (or Document.pdf for a read-only version).

