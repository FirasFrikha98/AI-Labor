#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
import time

base_url = "https://interweb.l3s.uni-hannover.de"
api_key = "Q655yutpcZEV4C1GrxAh2CGZz5qg2oAMX3QIa8q05MNjU8FNjXSmwh2XVbKKotqn"
url = f"{base_url}/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

input_file = "../input/tweet_keyword_mapping_push_factors_social_factors.json"
output_file = "../output/augmented_push_factors_social_factors.json"

label = "push_factors_social_factors"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

def call_api_with_retries(payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return response
            time.sleep(2) 
        except Exception as e:
            print(f"Error during API call (Attempt {attempt + 1}): {e}")
    return None

def generate_alternative_tweets(tweet):
    prompt = (
        f"Generate 5 alternative tweets that convey the same meaning as this one:\n\n"
        f"'{tweet}'\n\n"
        f"Output ONLY the 5 tweets as plain text, one tweet per line, without any additional text."
    )

    payload = {
        "model": "llama3.3:70b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 150
    }

    response = call_api_with_retries(payload)
    if response and response.status_code == 200:
        try:
            llm_output = response.json()["choices"][0]["message"]["content"]
            tweets = [line.strip() for line in llm_output.split("\n") if line.strip()]
            return tweets if len(tweets) == 5 else []
        except Exception:
            return []
    return []

def extract_rationale(tweet):
    prompt = (
        f"Extract the most relevant keywords or phrases from this tweet:\n\n"
        f"'{tweet}'\n\n"
        f"Output ONLY as a JSON array of strings."
    )

    payload = {
        "model": "llama3.3:70b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50
    }

    response = call_api_with_retries(payload)
    if response and response.status_code == 200:
        try:
            llm_output = response.json()["choices"][0]["message"]["content"]
            rationale = json.loads(llm_output)
            return rationale if isinstance(rationale, list) else []
        except Exception:
            return []
    return []

nested_augmented_data = []
success_count = 0
failure_count = 0

total_entries = len(data)

for idx, entry in enumerate(data, start=1):
    try:
        original_tweet = entry["original_tweet"]
        rationale = entry["keyword"]

        if isinstance(rationale, str):
            rationale = [word.strip() for word in rationale.split(",")]

        alternative_tweets = generate_alternative_tweets(original_tweet)
        augmentations = []

        if alternative_tweets:
            success_count += 1
            for aug_tweet in alternative_tweets:
                aug_rationale = extract_rationale(aug_tweet)
                augmentations.append({
                    "tweet": aug_tweet,
                    "rationale": aug_rationale
                })
            print(f"success {success_count}/{total_entries}")
        else:
            failure_count += 1
            print(f"failure {failure_count}/{total_entries}")

        nested_augmented_data.append({
            "tweet": original_tweet,
            "rationale": rationale,
            "label": label,
            "augmentations": augmentations
        })
    except Exception as e:
        failure_count += 1
        print(f"failure {failure_count}/{total_entries}")

try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(nested_augmented_data, f, indent=4)
    print(f"Augmentation complete. Saved to {output_file}.")
    print(f"Successes: {success_count}/{total_entries}, Failures: {failure_count}/{total_entries}")
except Exception as e:
    print(f"Error saving augmented data: {e}")


# In[ ]:




