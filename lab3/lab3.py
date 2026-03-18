import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

train_data = [
    ("Justice is finally served, the victims get support.", "Positive"),
    ("Truth revealed about the Epstein network, excellent work.", "Positive"),
    ("Програма захисту від расизму працює чудово, люди отримують підтримку.", "Positive"),
    ("Справедливість перемогла, дискримінацію зупинено.", "Positive"),
    ("Terrible crimes and abuse in the Epstein case.", "Negative"),
    ("It is an illegal and disgusting network of abuse.", "Negative"),
    ("Расизм та дискримінація – це жахливий злочин проти людяності.", "Negative"),
    ("Ганебне ставлення до людей через колір шкіри, це просто жах.", "Negative"),
    ("The court published the official document and report today.", "Neutral"),
    ("Fact-based statement regarding the Epstein island investigation.", "Neutral"),
    ("Офіційний звіт суду про випадки порушення прав у суспільстві.", "Neutral"),
    ("Статистична інформація та інформаційний звіт щодо подій.", "Neutral")
]

X_train = [text for text, label in train_data]
y_train = [label for text, label in train_data]

nlp_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
    ('classifier', LogisticRegression(random_state=42))
])

nlp_pipeline.fit(X_train, y_train)

test_reviews = [
    "The official court documents and reports were released to the public today.",
    "The abuse and illegal activities in the Epstein network are terrible.",
    "Excellent news! Justice is served for the victims of the network.",
    "We need more factual statements and documents about the investigation.",
    "Це просто жахливо, що така дискримінація та злочини досі існують.",
    "Чудова ініціатива! Люди отримують необхідну підтримку та захист.",
    "Опубліковано новий звіт та статистичну інформацію щодо звернень до суду.",
    "Ганебне явище расизму має бути повністю викорінене із суспільства.",
    "Звіт суду показує жахливі злочини, але чудово, що справедливість перемогла."
]

predictions = nlp_pipeline.predict(test_reviews)

probabilities = nlp_pipeline.predict_proba(test_reviews)
classes = nlp_pipeline.classes_


for i in range(len(test_reviews)):
    text = test_reviews[i]
    pred = predictions[i]

    max_prob = np.max(probabilities[i]) * 100

    short_text = text if len(text) < 53 else text[:50] + "..."

    print(f"{short_text:<55} | {pred:<10} | {max_prob:.1f}%")

