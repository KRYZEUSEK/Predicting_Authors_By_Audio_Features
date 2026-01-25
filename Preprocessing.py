import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

path = "C:/Users/niemi/.cache/kagglehub/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres/versions/1/songs.csv"
dataset = pd.read_csv(path)
dataset = dataset[dataset["artists"].duplicated(keep=False)]

artist_count = dataset["artists"].value_counts()
dataset["artist_frequency"] = dataset["artists"].map(artist_count)
dataset = dataset[dataset["artist_frequency"] > 200]
dataset = dataset.drop("artist_frequency", axis=1)
print(dataset.head())
print(dataset.shape)
print(dataset.dtypes)
print(dataset[dataset.duplicated()])
print(dataset.isna().sum())

label_encoder = LabelEncoder()

labels = dataset.loc[:, "artists"]

labels_encoded = label_encoder.fit_transform(labels)

labels_encoded = pd.DataFrame(labels_encoded, columns=["artists"], dtype=float).reset_index(drop=True)

print(type(labels_encoded))

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
].reset_index(drop=True)

label_encoder_genre = LabelEncoder()

genre = features[:]["genre"]
features_encoded = label_encoder_genre.fit_transform(genre)

features_encoded = pd.DataFrame(features_encoded, columns=["genre"], dtype=float).reset_index(drop=True)
features.drop("genre", axis=1, inplace=True)
features_pd = pd.concat([features, features_encoded], axis=1)
print(features_pd.head())
print(type(features_pd))

new_dataset = pd.concat([features_pd, labels_encoded], axis=1)

assert not new_dataset.isna().any().any()
print(new_dataset.head())
print(new_dataset.dtypes)
print(len(new_dataset.index), len(new_dataset.columns))
print(new_dataset["artists"].nunique())

Train, test = train_test_split(
    new_dataset, test_size=0.2, random_state=42, shuffle=True
)
print(Train.head(), test.head())
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
