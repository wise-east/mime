# load human_eval.jsonl and anonymize the results by encrypting the annotator field, accessed by ['url']['annotator']

import json 
import hashlib 

with open("results/mime-real-resized/human_eval.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

for d in data:
    d['url_data']["annotator"] = hashlib.sha256(d["url_data"]["annotator"].encode()).hexdigest()

with open("results/mime-real-resized/human_eval_anonymized.jsonl", "w") as f:  
    for d in data:
        f.write(json.dumps(d) + "\n")