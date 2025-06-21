import json 

fp = "raw_ff_gemini_gemini-1.5-flash.json"

with open(fp, "r") as f:
    data = json.load(f)

blanks = data[::3]
aligned = data[1::3]
misaligned = data[2::3]

# failure mode statistics
def compute_failure_mode_stats(data):
    # count the number of items in data with each failure mode and group the samples with same failure mode together
    failure_mode_counts = {}
    for item in data:
        failure_mode = item["fail_mode"]
        if failure_mode not in failure_mode_counts:
            failure_mode_counts[failure_mode] = []
        failure_mode_counts[failure_mode].append(item)
        
    for failure_mode, items in sorted(failure_mode_counts.items(), key=lambda x: x[0]):
        print(f"{failure_mode:<60}: {len(items):<2} ({len(items) / len(data) * 100:>5.1f}%)")
        
    print("---" * 10)
    return failure_mode_counts


blanks_stats = compute_failure_mode_stats(blanks)
aligned_stats = compute_failure_mode_stats(aligned)
misaligned_stats = compute_failure_mode_stats(misaligned)

results = {
    "blanks": blanks_stats,
    "aligned": aligned_stats,
    "misaligned": misaligned_stats,
}

# print as unified table 
all_keys = set(blanks_stats.keys()) | set(aligned_stats.keys()) | set(misaligned_stats.keys())

# pretty print table 
# columns = blanks, aligned, misaligned
# rows = all_keys
# print the table with the keys as the first row and the values as the second row
print(f"{'':<60} {'blanks':>18} {'aligned':>18} {'misaligned':>18}")
for key in all_keys:
    print(f"{key:<60}", end="")
    for dataset in results.values():
        if key in dataset:
            print(f"{len(dataset[key]):>10} ({len(dataset[key]) / len(blanks) * 100:>5.1f}%)", end="")
        else:
            empty_str = "-"
            print(f"{empty_str:>19}", end="")
    print()