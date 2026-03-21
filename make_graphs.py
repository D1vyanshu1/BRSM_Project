import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
import os

sns.set(style="whitegrid")

# consistent colors
MEAN_COLOR = "red"
MEDIAN_COLOR = "yellow"

# create plots directory
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# load data
vft = pd.read_csv("vft_data.csv")
spam = pd.read_csv("spam_coordinates.csv")

# ----------------------------------
# WORD COUNT PER PARTICIPANT
# ----------------------------------

word_counts = vft.groupby(['participant_id','domain']).size().reset_index(name='word_count')

plt.figure(figsize=(8,6))
sns.histplot(word_counts['word_count'], bins=15, kde=True)

plt.axvline(word_counts['word_count'].mean(), color=MEAN_COLOR, label="Mean")
plt.axvline(word_counts['word_count'].median(), color=MEDIAN_COLOR, label="Median")

plt.title("Distribution of Verbal Fluency Scores")
plt.xlabel("Number of Words Generated")
plt.ylabel("Frequency")
plt.legend()

plt.savefig(f"{PLOT_DIR}/01_fluency_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------------
# BOX PLOT CATEGORY FLUENCY
# ----------------------------------

plt.figure(figsize=(8,6))
sns.boxplot(data=word_counts, x='domain', y='word_count')

plt.axhline(word_counts['word_count'].mean(), color=MEAN_COLOR, label="Mean")

plt.title("Fluency Across Semantic Categories")
plt.xlabel("Category")
plt.ylabel("Number of Words")
plt.legend()

plt.savefig(f"{PLOT_DIR}/02_category_fluency_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------------
# IRT DISTRIBUTION
# ----------------------------------

plt.figure(figsize=(8,6))
sns.histplot(vft['IRT'], bins=20, kde=True)

plt.axvline(vft['IRT'].mean(), color=MEAN_COLOR, label="Mean")
plt.axvline(vft['IRT'].median(), color=MEDIAN_COLOR, label="Median")

plt.title("Distribution of Inter Response Time")
plt.xlabel("IRT (seconds)")
plt.ylabel("Frequency")
plt.legend()

plt.savefig(f"{PLOT_DIR}/03_irt_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------------
# IRT VS WORD ORDER
# ----------------------------------

plt.figure(figsize=(8,6))
sns.lineplot(data=vft, x='word_order', y='IRT', estimator='mean')

plt.axhline(vft['IRT'].mean(), color=MEAN_COLOR, label="Mean IRT")

plt.title("Retrieval Difficulty Over Time")
plt.xlabel("Word Order")
plt.ylabel("Mean IRT")
plt.legend()

plt.savefig(f"{PLOT_DIR}/04_irt_vs_word_order.png", dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------------
# MEAN FLUENCY PER CATEGORY
# ----------------------------------

mean_cat = word_counts.groupby('domain')['word_count'].mean().reset_index()

plt.figure(figsize=(8,6))
sns.barplot(data=mean_cat, x='domain', y='word_count')

plt.axhline(mean_cat['word_count'].mean(), color=MEAN_COLOR, label="Overall Mean")

plt.title("Average Fluency by Category")
plt.xlabel("Category")
plt.ylabel("Mean Word Count")
plt.legend()

plt.savefig(f"{PLOT_DIR}/05_mean_fluency_by_category.png", dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------------
# MOST FREQUENT WORDS
# ----------------------------------

top_words = vft['word'].value_counts().head(15)

plt.figure(figsize=(10,6))
sns.barplot(x=top_words.values, y=top_words.index)

plt.title("Most Frequently Generated Words")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.legend(["Word Frequency"])

plt.savefig(f"{PLOT_DIR}/06_top_words_frequency.png", dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------------
# SEMANTIC MAP
# ----------------------------------

plt.figure(figsize=(8,8))
sns.scatterplot(data=spam, x='x_norm', y='y_norm', hue='domain')

for i,row in spam.iterrows():
    plt.text(row['x_norm'], row['y_norm'], row['word'], fontsize=8)

plt.title("Semantic Spatial Organization")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.legend()

plt.savefig(f"{PLOT_DIR}/07_semantic_map.png", dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------------
# SEMANTIC DISTANCE DISTRIBUTION
# ----------------------------------

coords = spam[['x_norm','y_norm']]
distances = pdist(coords)

plt.figure(figsize=(8,6))
sns.histplot(distances, bins=20)

plt.axvline(np.mean(distances), color=MEAN_COLOR, label="Mean Distance")

plt.title("Distribution of Semantic Distances")
plt.xlabel("Pairwise Distance")
plt.ylabel("Frequency")
plt.legend()

plt.savefig(f"{PLOT_DIR}/08_semantic_distance_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nAll plots saved in '{PLOT_DIR}/' directory.")