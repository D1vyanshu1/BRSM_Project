import json
import pandas as pd

# Load JSON
with open("responses.json", "r") as f:
    data = json.load(f)

experiment = data["fluency-spam"]

vft_rows = []
spam_rows = []

# Iterate through all participants
for session_id, session_data in experiment.items():

    participant_id = session_data["subject_id"]
    trials = session_data["data"]

    for trial in trials:

        task = trial.get("task")
        domain = trial.get("domain")

        # Remove practice trials
        if domain == "furniture-practice":
            continue

        # -------------------------
        # VFT TASK (Word generation)
        # -------------------------
        if task == "VFT":

            responses = trial.get("tagged_responses")
            times = trial.get("response_times")

            if responses and times:

                responses = json.loads(responses)
                times = json.loads(times)

                for i, r in enumerate(responses):

                    word = r["response"]
                    irt = times[i] / 1000  # convert ms → seconds

                    vft_rows.append({
                        "participant_id": participant_id,
                        "domain": domain,
                        "word_order": i + 1,
                        "word": word,
                        "IRT": irt
                    })

        # -------------------------
        # SpAM TASK (Spatial layout)
        # -------------------------
        if task == "SpAM":

            dropped = trial.get("droppedwords")

            if dropped:

                for item in dropped:

                    spam_rows.append({
                        "participant_id": participant_id,
                        "domain": domain,
                        "word": item["word"],
                        "x_norm": item["x_norm"],
                        "y_norm": item["y_norm"]
                    })


# Convert to DataFrames
vft_df = pd.DataFrame(vft_rows)
spam_df = pd.DataFrame(spam_rows)

# Keep only the final position of each word in SpAM
spam_df = spam_df.groupby(
    ["participant_id", "domain", "word"]
).last().reset_index()

# Save CSV files
vft_df.to_csv("vft_data.csv", index=False)
spam_df.to_csv("spam_coordinates.csv", index=False)

print("CSV files generated successfully!")
print("VFT rows:", len(vft_df))
print("SpAM rows:", len(spam_df))
print("Participants:", vft_df["participant_id"].nunique())