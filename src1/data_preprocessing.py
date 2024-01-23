import pandas as pd

def read_data(train_X_path, train_y_path):
    data = pd.read_csv(train_X_path)
    target = pd.read_csv(train_y_path)
    
    # data = pd.read_csv(config['train_path'])
    # target = pd.read_csv(config['target_path'])
    return data, target

def inspect_data(df):
    print(
        df.columns,
        df.info(),
        df.describe(),
        df.head(10),
        df.nunique(),
        df.columns[df.isna().any()].tolist(),
        sep = '\n'
        )

if config.get('eda_on'):
    inspect_data(data)
    inspect_data(target)

if config.get('drop_cols'):
    y = target.drop(columns=config['drop_cols'])
    X = data.drop(columns=config['drop_cols'])
else:
    y = target
    X = data

# Create a plotting fxn
#sns.pairplot(X)
#X.hist(figsize=(15, 10))

#ss = StandardScaler().set_output(transform="pandas")
#X_scaled = ss.fit_transform(X)

# Create a plotting fxn
#fig, ax = plt.subplots(figsize=(18,8))
#sns.boxplot(ax=ax, data=X_scaled, orient='v')
#plt.xticks(rotation=45);

# More EDA
#X.select_dtypes(include=["object"]).apply(lambda col: len(col.unique()))