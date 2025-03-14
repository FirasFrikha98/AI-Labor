import os
import json
import pandas as pd

keywords_dir = "../training_data/keywords/"  
augmented_dir = "../output/" 
output_dirs = {
    "first": "../training_data/5x/first",
    "middle": "../training_data/5x/middle",
    "last": "../training_data/5x/last"
}

for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)

def split_keywords_into_parts(keywords):
    total = len(keywords)
    split_size = total // 3
    first_30 = keywords[:split_size]
    middle_30 = keywords[split_size: 2 * split_size]
    last_30 = keywords[2 * split_size:]
    return first_30, middle_30, last_30

def process_class(keyword_file, augmented_file, label):
    with open(keyword_file, "r", encoding="utf-8") as f:
        keywords = [line.strip() for line in f if line.strip()]

    with open(augmented_file, "r", encoding="utf-8") as f:
        augmented_data = json.load(f)

    original_tweets = []
    augmentations_1st = []
    augmentations_2nd = []
    augmentations_3rd = []
    augmentations_4th = []
    augmentations_5th = []
    for entry in augmented_data:
        if "augmentations" in entry:
            original_tweets.append({
                "tweet": entry["tweet"],
                "rationale": entry["rationale"],
                "label": label
            })
            if len(entry["augmentations"]) > 0:
                augmentations_1st.append({
                    "tweet": entry["augmentations"][0]["tweet"],
                    "rationale": entry["augmentations"][0]["rationale"],
                    "label": label
                })
            if len(entry["augmentations"]) > 1:
                augmentations_2nd.append({
                    "tweet": entry["augmentations"][1]["tweet"],
                    "rationale": entry["augmentations"][1]["rationale"],
                    "label": label
                })
            if len(entry["augmentations"]) > 2:
                augmentations_3rd.append({
                    "tweet": entry["augmentations"][2]["tweet"],
                    "rationale": entry["augmentations"][2]["rationale"],
                    "label": label
                })
            if len(entry["augmentations"]) > 3:
                augmentations_4th.append({
                    "tweet": entry["augmentations"][3]["tweet"],
                    "rationale": entry["augmentations"][3]["rationale"],
                    "label": label
                })
            if len(entry["augmentations"]) > 4:
                augmentations_5th.append({
                    "tweet": entry["augmentations"][4]["tweet"],
                    "rationale": entry["augmentations"][4]["rationale"],
                    "label": label
                })

    first_30, middle_30, last_30 = split_keywords_into_parts(keywords)

    datasets = {
        "first": {"training": [], "validation": []},
        "middle": {"training": [], "validation": []},
        "last": {"training": [], "validation": []}
    }
    for split_name, split_keywords in zip(["first", "middle", "last"], [first_30, middle_30, last_30]):
        for idx, entry in enumerate(original_tweets):
            if any(keyword in entry["rationale"] for keyword in split_keywords):
                datasets[split_name]["validation"].append(entry)
            else:
                datasets[split_name]["training"].append(entry)
                if idx < len(augmentations_1st):
                    datasets[split_name]["training"].append(augmentations_1st[idx])
                if idx < len(augmentations_2nd):
                    datasets[split_name]["training"].append(augmentations_2nd[idx])
                if idx < len(augmentations_3rd):
                    datasets[split_name]["training"].append(augmentations_3rd[idx])
                if idx < len(augmentations_4th):  
                    datasets[split_name]["training"].append(augmentations_4th[idx])
                if idx < len(augmentations_5th):   
                    datasets[split_name]["training"].append(augmentations_5th[idx])

    return datasets, {
        "Class": label,
        "Total Keywords": len(keywords),
        "Total Tweets": len(original_tweets),
        "First 30% Matches": len(datasets["first"]["validation"]),
        "Middle 30% Matches": len(datasets["middle"]["validation"]),
        "Last 30% Matches": len(datasets["last"]["validation"]),
    }

combined_datasets = {
    "first": {"training": [], "validation": []},
    "middle": {"training": [], "validation": []},
    "last": {"training": [], "validation": []}
}
class_stats = []
classes = [
    ("pull_factors_economy.txt", "augmented_pull_factors_economy.json", "pull_factors_economy"),
    ("pull_factors_environment_health.txt", "augmented_pull_factors_environment_health.json", "pull_factors_environment_health"),
    ("pull_factors_political_factors.txt", "augmented_pull_factors_political_factors.json", "pull_factors_political_factors"),
    ("pull_factors_social_factors.txt", "augmented_pull_factors_social_factors.json", "pull_factors_social_factors"),
    ("push_factors_conflict.txt", "augmented_push_factors_conflict.json", "push_factors_conflict"),
    ("push_factors_economy.txt", "augmented_push_factors_economy.json", "push_factors_economy"),
    ("push_factors_environment.txt", "augmented_push_factors_environment.json", "push_factors_environment"),
    ("push_factors_health.txt", "augmented_push_factors_health.json", "push_factors_health"),
    ("push_factors_political.txt", "augmented_push_factors_political.json", "push_factors_political"),
    ("push_factors_social_factors.txt", "augmented_push_factors_social_factors.json", "push_factors_social_factors"),
]

for keyword_file, augmented_file, label in classes:
    keyword_path = os.path.join(keywords_dir, keyword_file)
    augmented_path = os.path.join(augmented_dir, augmented_file)
    class_datasets, stats = process_class(keyword_path, augmented_path, label)

    for split_name in ["first", "middle", "last"]:
        combined_datasets[split_name]["training"].extend(class_datasets[split_name]["training"])
        combined_datasets[split_name]["validation"].extend(class_datasets[split_name]["validation"])

    class_stats.append(stats)

for split_name, datasets in combined_datasets.items():
    training_path = os.path.join(output_dirs[split_name], "training.json")
    validation_path = os.path.join(output_dirs[split_name], "validation.json")

    with open(training_path, "w", encoding="utf-8") as f:
        json.dump(datasets["training"], f, indent=4)

    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(datasets["validation"], f, indent=4)
stats_df = pd.DataFrame(class_stats)
stats_df.columns = [
    "Class",
    "Total Keywords",
    "Total Tweets",
    "First 30% Matches",
    "Middle 30% Matches",
    "Last 30% Matches",
]
print("\n=== Class Statistics Table ===")
print(stats_df.to_string(index=False))

