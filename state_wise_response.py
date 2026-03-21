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

# merge with participant data
merged = participants.merge(dominant_language, on="participant_id")

# count participants per state and language
state_language = (
    merged.groupby(["state","language"])
    .size()
    .reset_index(name="count")
)

print(state_language)

# plot
plt.figure(figsize=(10,6))

sns.barplot(
    data=state_language,
    x="state",
    y="count",
    hue="language"
)

plt.title("Participant Language Preference by State")
plt.xlabel("State")
plt.ylabel("Number of Participants")
plt.xticks(rotation=45)
plt.legend(title="Language")

plt.tight_layout()
plt.show()