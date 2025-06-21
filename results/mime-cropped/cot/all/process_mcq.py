from pathlib import Path 
import json 
import pandas as pd 

def load_json(fp):
    with open(fp, "r") as f:
        return json.load(f)

for file in Path(__file__).parent.glob("*mcq*.json"):
    data = load_json(file)
    preds ={} 
    for item in data["predictions"]:
        preds[item["instance_id"]] = item["is_correct"]
        
        
    # check that the results are the same as those in the score files 
    score_file = file.replace("raw", "score")
    score_data = load_json(score_file)
    has_faulty_results = False
    for item in score_data["predictions"]:
        if item["instance_id"] not in preds:
            print(f"Instance ID {item['instance_id']} not found in {file}")
            has_faulty_results = True
        elif preds[item["instance_id"]] != item["is_correct"]:
            print(f"Instance ID {item['instance_id']} has incorrect result in {file}")
            has_faulty_results = True
            
    if has_faulty_results:
        print(f"There are faulty results in {file}")
        continue 
            
    # save the predictions 
    preds_df = pd.DataFrame(preds, index=[0])
    preds_df.to_csv(file.replace("raw", "score").replace(".json", ".csv"), index=False)
    
    
    
    