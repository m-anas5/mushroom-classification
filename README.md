## Mushroom Classification Using Machine Learning: Edible vs Poisnous

This project applies machine learning algorithms to classify mushrooms as edible or poisnous based on their physical characteristics. This dataset was taken from Kaggle [link](https://www.kaggle.com/datasets/uciml/mushroom-classification)


## Overview

  The task is to use 22 categorical features describing physical characteristics of mushrooms, such as cap shape, color, gill size and spore print color to predict whether a mushroom is edible or poisnous (target column class). The dataset originally is sourced from UCI Machine Learning Repository, contains labeled examples with one column (stalk-root) missing 2480 values. Since this has a target column, it makes it well suited for supervised learning. The primary challenge lies in accurately modeling the relationships between categorical variables and ensuring the model generalizes well to unseen mushroom types.

  The approach taken in this project formulates the problem as a binary classification task, aiming to predict whether a mushroom is edible or poisonous. Initial experiments were conducted using Logistic Regression and Decision Tree classifiers to evaluate performance on the dataset. After observing promising results with the Decision Tree model, cross-validation was applied using 4 splits to better assess its generalization and reduce the risk of overfitting. This comparative approach allowed for evaluating the effectiveness and interpretability of both linear and tree-based methods on categorical data.

  Both the Logistic Regression and Decision Tree models achieved perfect classification accuracy of 100% on the mushroom dataset, correctly identifying all edible and poisonous mushrooms. Cross-validation using 4 folds with the Decision Tree Classifier confirmed this robustness, consistently yielding 1.0 accuracy, precision, recall, and F1-score across all folds. To evaluate generalization with constrained complexity, a Decision Tree with tuned hyperparameters (criterion='entropy', max_depth=3) was also tested, achieving a strong accuracy of 96.25%, indicating that even a shallow tree can perform well on this dataset while offering greater interpretability.

## Summary of Workdone

Include only the sections that are relevant an appropriate.

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

![Plot of each column including target](/1.png)
