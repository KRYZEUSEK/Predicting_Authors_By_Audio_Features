import kagglehub
import pandas as pd

dataset = pd.read_csv("C:/Users/niemi/.cache/kagglehub/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres/versions/1/songs.csv")
print(dataset.info())
print(dataset.describe())

labels = dataset[:]["artists"]
f
