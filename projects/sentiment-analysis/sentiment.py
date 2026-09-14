from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "This product is excellent",
    "I had a terrible experience",
]
labels = [1, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

prediction = model.predict(vectorizer.transform(["The experience was excellent"]))
print("Prediction:", prediction[0])
