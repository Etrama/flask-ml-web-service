import pandas as pd
import config
from sklearn.ensemble import RandomForestClassifier
import joblib
# training a simple sklearn model
df = pd.read_csv(config.processed_data)
x = df[df.columns.difference(["Survived"])]
y = df["Survived"]
classifier = RandomForestClassifier()
classifier.fit(x, y)
print("Training complete.")
# persisting the model to a file
joblib.dump(classifier, config.model_store + "RandomForestDefault.pkl")