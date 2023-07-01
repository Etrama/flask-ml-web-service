import pandas as pd
import config
# we just read the file
# format it in a ML-able format, rough approach here
# just to get my hands dirsty with Flask
df = pd.read_csv(config.train_data)
df = df[config.numerical_cols]
df = df.dropna()
df = df.reset_index(drop=True)
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
# apparently using k-1 dummies for k categories is better. When? I guess all the time? ;)
df.to_csv(config.processed_data, index=False)