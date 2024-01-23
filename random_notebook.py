import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from pycaret.utils import version
from pycaret.classification import *

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

######
# Option 1 #
from src1 import data_preprocessing as dataprep
dataprep.read_data()
dataprep.inspect_data()
######
# Option 2 #
from src1.data_preprocessing import read_data, inspect_data
data, target = read_data(config['train_path'], config['target_path'])
inspect_data(data)
#####

with open('run_config.yml') as f:
    config = yaml.load(f)



# data = pd.read_csv(config['train_path'])

# target = pd.read_csv(config['target_path'])

# def inspect_data(df):
#     print(
#         df.columns,
#         df.info(),
#         df.describe(),
#         df.head(10),
#         df.nunique(),
#         df.columns[df.isna().any()].tolist(),
#         sep = '\n'
#         )

# if config.get('eda_on'):
#     inspect_data(data)
#     inspect_data(target)

# if config.get('drop_cols'):
#     y = target.drop(columns=config['drop_cols'])
#     X = data.drop(columns=config['drop_cols'])
# else:
#     y = target
#     X = data

# # Create a plotting fxn
# #sns.pairplot(X)
# #X.hist(figsize=(15, 10))

# ss = StandardScaler().set_output(transform="pandas")
# X_scaled = ss.fit_transform(X)

# # Create a plotting fxn
# #fig, ax = plt.subplots(figsize=(18,8))
# #sns.boxplot(ax=ax, data=X_scaled, orient='v')
# #plt.xticks(rotation=45);

# # More EDA
# #X.select_dtypes(include=["object"]).apply(lambda col: len(col.unique()))

"""## Build a Baseline Model with Raw Features

### PyCaret
"""

#!pip install --pre pycaret
#!pip install "schemdraw<0.16" #<-- To handle dependency issues

# Run this for more advanced tuning strategies
# check documentation: https://pycaret.readthedocs.io/en/latest/api/classification.html#pycaret.classification.ClassificationExperiment.tune_model
#!pip install pycaret[tuners]

# for some model interpretation functions
#!pip install interpret



clf = setup(data=pd.concat([X, y], axis=1),
            target='Expected',
            session_id=13,
            experiment_name='kaggle_constructor',
            n_jobs=-1)

best_model = compare_models(fold=5, sort='F1')

best_model

lightgbm = create_model('lightgbm', fold=5)

tuned_lightgbm = tune_model(lightgbm,
                      n_iter=10,
                      optimize='f1',
                      fold=5)

plot_model(tuned_lightgbm, plot='feature')

final_lightgbm = finalize_model(tuned_lightgbm)
final_lightgbm

save_model(final_lightgbm, model_name='final_lightgbm')

#hardcoded
test_path = '/content/drive/My Drive/Constructor_Academy/Gitlab/justin-villard/04_MachineLearning/day6/features_test.csv'
test = pd.read_csv(test_path)
test.head(10)

X_test = test.drop(columns=['Id'])
X_test

prediction_df = predict_model(final_lightgbm, data=X_test)
prediction_df.head()

y_test = pd.DataFrame({'Id': np.arange(0,len(prediction_df['prediction_label'])),
                       'Predicted': prediction_df['prediction_label']
                      })
y_test

#hardcoded
y_test.to_csv('/content/drive/MyDrive/Constructor_Academy/Gitlab/justin-villard/04_MachineLearning/day6/final_lightgbm.csv', index=False)

"""### Select data in the interquartile range, impute NaN values

"""

def filter_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column[(column >= lower_bound) & (column <= upper_bound)]

filtered_X = X.apply(filter_iqr) #.dropna()
filtered_X

clf = setup(data=pd.concat([filtered_X, y], axis=1),
            target='Expected',
            numeric_imputation='knn',
            session_id=1,
            experiment_name='kaggle_constructor_iqr',
            n_jobs=-1)

best_model_filtered = compare_models(fold=5, sort='F1')

best_model_filtered

filtered_lightgbm = finalize_model(best_model_filtered)
filtered_lightgbm

save_model(filtered_lightgbm, model_name='filtered_lightgbm')

prediction_df = predict_model(filtered_lightgbm, data=X_test)
prediction_df.head()

"""Last attempt with the chosen model"""


# Create a pipeline with a StandardScaler and a LightGBM classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),   # StandardScaler to scale numeric data
    ('lgbm', LGBMClassifier())      # LightGBM Classifier
])

# Perform 5-fold cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1-macro')

# Fit the pipeline on the training data (optional if you want to use the cross_val_score results)
pipeline.fit(X, y)

# Evaluate the model using the cross-validated scores
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {np.mean(cv_scores)}')

# Predict on the test data
y_pred = pipeline.predict(X_test)

