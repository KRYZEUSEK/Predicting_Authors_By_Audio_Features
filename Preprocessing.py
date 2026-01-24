import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset

path = "C:/Users/niemi/.cache/kagglehub/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres/versions/1/songs.csv"
dataset = pd.read_csv(path)
#print(dataset.head())
#print(dataset.shape)
#print(dataset.dtypes)
#print(dataset[dataset.duplicated()])
#print(dataset.isna().sum())

label_encoder = LabelEncoder()

labels = dataset.loc[:, "artists"]
labels_encoded = label_encoder.fit_transform(labels)

labels_pd = pd.DataFrame(labels_encoded, columns=["artists"], dtype=float)
print(type(labels_pd))

features = dataset[:][
    [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
        "genre",
    ]
]
genre = features[:]["genre"]

features_encoded = label_encoder.fit_transform(genre)
genre_pd = pd.DataFrame(features_encoded, dtype=float)

features = features.drop("genre", axis=1)
features_pd = pd.concat([features, genre_pd], axis=1)
print(features_pd.head())
print(type(features_pd))

new_dataset = pd.concat([features_pd, labels_pd], axis=1)

print(new_dataset.head())
print(len(new_dataset.index), len(new_dataset.columns))
print(new_dataset["artists"].nunique())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Train, test = train_test_split(
    new_dataset, test_size=0.2, random_state=42, shuffle=True
)
train_features = Train.iloc[:, :-1].values
train_labels = Train.iloc[:, -1].values

test_features = test.iloc[:, :-1].values
test_labels = test.iloc[:, -1].values

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

Train.iloc[:, :-1] = train_features
test.iloc[:, :-1] = test_features

print(Train.dtypes, test.dtypes)
print(len(Train.index), len(test.index))