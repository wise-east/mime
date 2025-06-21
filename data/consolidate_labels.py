# This script is used to consolidate the labels from the annotations. 
# It is used to create the consolidated_data.jsonl file. 
# It is also used to manually examine the consolidated_data.jsonl file and fix them such that the final label contains the minimum amount of terms, for synonyms go from most specific to least specific 

import pandas as pd 
from pathlib import Path 
import json
from tqdm import tqdm
from MimeEval.utils.constants import PACKAGE_DIR

# aggregate jsonl results
with open(PACKAGE_DIR / "data" / "initial_annotations.jsonl", "r") as f: 
    data = [json.loads(line) for line in f]

# aggregate on sample_id so that user_answer is a list of answers
data = pd.DataFrame(data)
data = data.groupby("sample_id").agg(list).reset_index()

agg_data = {}
# just keep the first question for each sample_id 
for index, row in data.iterrows():  
    # breakpoint() 
    question_data = row["question"][0]
    annotators = [item["prolific_id"] for item in row["url_data"]] 
    
    agg_data[row["sample_id"]] = {
        "annotators": annotators,
        "answers": row["user_answer"],
        "label": question_data["label"],
        "avatar": question_data["avatar"],
        "action": question_data["action"],
        "background": question_data["background"],
        "url": question_data["s3_url"]
    }

with open(PACKAGE_DIR / "data" / "mime_data_legacy.jsonl", "r") as f: 
    unconsolidated_data = [json.loads(line) for line in f]

consolidated_datapath = PACKAGE_DIR / "data" / "consolidated_data.jsonl"
if not Path(consolidated_datapath).exists(): 
    consolidated_data = []
else: 
    with open(consolidated_datapath, "r") as f: 
        consolidated_data = [json.loads(line) for line in f]

completed_sample_ids = [sample["sample_id"] for sample in consolidated_data]
for sample in tqdm(unconsolidated_data):  
    if sample["sample_id"] in completed_sample_ids: 
        continue 
    
    agg_result = agg_data[sample["sample_id"]]
    
    agg_answers = []
    for answer in agg_result["answers"]: 
        answer_set = answer.split(",")
        for _answer in answer_set: 
            _answer = _answer.lower().strip()
            if _answer not in agg_answers: 
                agg_answers.append(_answer)
    
    print("\n\n--------------------------------")
    print(agg_result["annotators"])
    print(agg_result["answers"])
    print(agg_answers)
    print(agg_result["label"])
    print(agg_result["avatar"])
    print(agg_result["url"])
    
    is_pass = input("Is this a pass? (y/n) Or use label? (u) ")
    if is_pass == "y": 
        sample["final_label"] = agg_answers
        with open(consolidated_datapath, "a") as f: 
            f.write(json.dumps(sample) + "\n")
    elif is_pass == "u": 
        sample["final_label"] = agg_result["label"]
        with open(consolidated_datapath, "a") as f: 
            f.write(json.dumps(sample) + "\n")
    else: 
        enter_new_label = input("Enter new label? (y/n) ")
        if enter_new_label == "y": 
            new_labels = input("Enter new labels (split by comma): ")
            sample["final_label"] = [label.lower().strip() for label in new_labels.split(",")]
            print(f"New labels: {sample['final_label']}")
            with open(consolidated_datapath, "a") as f: 
                f.write(json.dumps(sample) + "\n")
        else: 
            print("Skipping for now...")
            continue 
    
    
# load consolidated data 
with open(consolidated_datapath, "r") as f: 
    consolidated_data = [json.loads(line) for line in f]

# make sure that all final labels are in the form of a list
for sample in consolidated_data: 
    if isinstance(sample["final_label"], str): 
        sample["final_label"] = [sample["final_label"]]
        
# sort by action 
consolidated_data = sorted(consolidated_data, key=lambda x: x["action"])
        
with open(consolidated_datapath, "w") as f: 
    for sample in consolidated_data: 
        f.write(json.dumps(sample) + "\n")


# manually examine consolidated_data.jsonl and fix them such that the final label contains the minimum amount of terms, for synonyms go from most specific to least specific 
# --- done manually --- 

    

