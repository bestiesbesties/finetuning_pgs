import os
import pandas as pd

df = pd.read_csv(os.path.join("data", "parsed-pgs33.csv" ))
print("len(df): ", len(df))

## chars
df["chars_length"] = df["text"].str.len()
chars_median = int(df["chars_length"].median())
print("chars_median: ", chars_median)

## words
df["words_length"] = df["text"].str.split().str.len()
words_median = int(df["words_length"].median())
print("words_median: ", words_median)

## filter 
df = df[df["words_length"] <= words_median * 2]
print("len(df): ", len(df))
df.drop(["chars_length", "words_length"], axis=1, inplace=True)
df.to_csv(os.path.join("data", "parsed-pgs33_pruned.csv"))
