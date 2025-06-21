# split data into train / val / test 

import json 
import pandas as pd 
from pathlib import Path  
from loguru import logger 
from MimeEval.utils.constants import PACKAGE_DIR

with open(PACKAGE_DIR / "data" / "mime_data_legacy.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

### Split data to train / val / test  
# 60% train, 20% val, 20% test 
# only keep the samples with background as blank 

# data = [d for d in data if d["background_config"] == "blank"]

# convert to pandas dataframe 
df = pd.DataFrame(data)

# get all the unique actions 
actions = df["action"].unique()

# print actions 
logger.info(sorted(actions))
logger.info(len(actions))
training_actions = [
    "Archery001", "Archery01",
    "ArmCurls001", "ArmCurls01", "ArmCurls03",
    "Baseball004", "BaseballPitch002", "BaseballPitch02",
    "Basketball001", "BasketballLayup001", "BasketballLayup02", "BasketballShot02",
    "Bowling003", "Bowling01",
    "Boxing001", "Boxing03",
    "CheckingPhone002",
    "ShootingAHandgun001", "ShootingARifle001", "ShootingHandgun04",
    "ShotPut001", "ShotPut01",
    "SittingAndWriting001",
    "Soccer003", "SoccerShot01",
    "Swimming001", "Swimming002", "Swimming03", "Swimming04", "Swimming06",
    "TakingPhotoWithCamera001",
    "Violin002",
    "WeightedSquat002",
    "Weightlifting001",
    "ConsoleGaming01"
]

val_actions = [
    "CheckingWatch001", "CheckingWatch01",
    "Climbing001", "Climbing01",
    "Darts001",
    "DeadLift001", "Deadlift01",
    "DrinkingCoffee001",
    "Driving002", "Driving003",
    "DrivingSitting001", "DrivingSittingDown03",
    "Volleyball001", "VolleyballServe",
    "WatchingTV01"
]

test_actions = [
    "Fencing001", "Fencing01",
    "Fishing001",
    "GolfPutting001", "GolfSwing001",
    "Javelin001",
    "Keyboard001", "KeyboardTyping03",
    "KnockingOnADoor001",
    "OpeningADoor001", "OpeningADoor002", "OpeningADoor03",
    "Piano002",
    "PlayingAGuitar002", "PlayingAGuitar004",
    "PlayingDrums001", "PlayingDrums03",
    "PlayingGuitar02", "PlayingGuitar03",
    "PlayingHarp001", "PlayingHarp002", "PlayingHarp01",
    "PlayingViolin01",
    "Pulling001", "Pulling002", "Pulling01", "Pulling02",
    "Pushing001", "Pushing003", "Pushing02",
    "PuttingOnSeatbelt03",
    "Seatbelt001",
    "TennisServe001", "TennisServe02",
    "TennisSwing001", "TennisSwing02"
]

# print the actions that are not in any of the splits 
not_included = set(actions) - set(training_actions) - set(val_actions) - set(test_actions)
assert len(not_included) == 0, f"The following actions are not included in any of the splits: {not_included}"

# print the number of actions in each split 
logger.info(f"Training actions: {len(training_actions)}")
logger.info(f"Validation actions: {len(val_actions)}")
logger.info(f"Test actions: {len(test_actions)}")

# assign splits to data 
for d in data:
    if d["action"] in training_actions:
        d["split"] = "train"
    elif d["action"] in val_actions:
        d["split"] = "val"
    elif d["action"] in test_actions:
        d["split"] = "test"

# save to jsonl     
with open(PACKAGE_DIR / "data" / "mime_data_legacy_with_splits.jsonl", "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")

