import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# load data
participants = pd.read_csv("participants.csv")
vft = pd.read_csv("vft_data.csv")

# detect language
def detect_language(word):
    if re.search(r'[\u0900-\u097F]', str(word)):
        return "Hindi"
    else:
        return "English/Hinglish"

vft["language"] = vft["word"].apply(detect_language)

# dominant language per participant
dominant_language = (
    vft.groupby("participant_id")["language"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

# merge datasets
merged = participants.merge(dominant_language, on="participant_id")

print(merged[["participant_id","age","language"]])

# plot
plt.figure(figsize=(8,6))

sns.boxplot(
    data=merged,
    x="language",
    y="age"
)

plt.title("Participant Age Distribution by Response Language")
plt.xlabel("Response Language")
plt.ylabel("Age")

plt.show()