import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

# -----------------------------
# Create plots directory
# -----------------------------
PLOT_DIR = "plots_for_vft"
os.makedirs(PLOT_DIR, exist_ok=True)

# enforce order
language_order = ["English/Hinglish", "Hindi"]

# -----------------------------
# Load dataset
# -----------------------------
vft = pd.read_csv("vft_data.csv")

# -----------------------------
# Detect Hindi vs English/Hinglish
# -----------------------------
def detect_language(word):
    if re.search(r'[\u0900-\u097F]', str(word)):
        return "Hindi"
    else:
        return "English/Hinglish"

vft["language"] = vft["word"].apply(detect_language)

# -----------------------------
# Standard color palette
# -----------------------------
COLOR_MAP = {
    "Hindi": "#1f77b4",            # blue
    "English/Hinglish": "#ff7f0e"  # orange
}

sns.set_style("whitegrid")

# ------------------------------------------------
# 1. Plot: Number of Responses by Language
# ------------------------------------------------

response_counts = vft["language"].value_counts().reset_index()
response_counts.columns = ["language","count"]

plt.figure(figsize=(7,6))

sns.barplot(
    data=response_counts,
    x="language",
    y="count",
    palette=COLOR_MAP,
    order=language_order
)

plt.title("Number of Responses by Language")
plt.xlabel("Language")
plt.ylabel("Number of Responses")
plt.legend(title="Language")

for i,row in response_counts.iterrows():
    plt.text(i,row["count"]+5,str(row["count"]),ha="center")

plt.savefig(f"{PLOT_DIR}/01_responses_by_language.png",dpi=300,bbox_inches="tight")
plt.close()

# ------------------------------------------------
# 2. Plot: Participants by Dominant Language
# ------------------------------------------------

participant_language = (
    vft.groupby("participant_id")["language"]
    .agg(lambda x: x.value_counts().idxmax())
)

participant_counts = participant_language.value_counts().reset_index()
participant_counts.columns = ["language","count"]


plt.figure(figsize=(7,6))

sns.barplot(
    data=participant_counts,
    x="language",
    y="count",
    order=language_order,
    palette=COLOR_MAP
)

plt.title("Participants by Dominant Language")
plt.xlabel("Language")
plt.ylabel("Number of Participants")

for i,row in participant_counts.iterrows():
    plt.text(i,row["count"]+0.3,str(row["count"]),ha="center")

plt.legend(title="Language")

plt.savefig(f"{PLOT_DIR}/02_participants_by_language.png",dpi=300,bbox_inches="tight")
plt.close()

# ------------------------------------------------
# 3. Plot: Words per Participant by Language
# ------------------------------------------------

words_per_participant = (
    vft.groupby(["participant_id","language"])
    .size()
    .reset_index(name="word_count")
)

plt.figure(figsize=(10,6))

sns.boxplot(
    data=words_per_participant,
    x="language",
    y="word_count",
    palette=COLOR_MAP,
    order=language_order
)

# mean + median lines
mean_val = words_per_participant["word_count"].mean()
median_val = words_per_participant["word_count"].median()

plt.axhline(mean_val,color="red",label="Mean")
plt.axhline(median_val,color="gold",label="Median")

plt.title("Distribution of Word Counts per Participant by Language")
plt.xlabel("Language")
plt.ylabel("Number of Words")
plt.legend()

plt.savefig(f"{PLOT_DIR}/03_words_per_participant_language.png",dpi=300,bbox_inches="tight")
plt.close()

print(f"\nAll plots saved in '{PLOT_DIR}/' directory.")