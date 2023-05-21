import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load and preprocess the dataset
df = pd.read_csv("spam.csv", encoding='latin1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1": "label", "v2": "message"})
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label"]

# Perform TF-IDF vectorization on the message column
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train the KNN classifier
k = 5  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

# Save the trained model and vectorizer
joblib.dump(knn, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
