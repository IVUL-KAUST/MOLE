import json 
from glob import glob 

files = glob("evals/model/test/*.json")
max_bench = 0
benchmarks = set()
for file in files:
    data = json.load(open(file))
    # max_bench = max(max_bench, len(data['Benchmarks']))
    if max_bench >= 65:
        print(file)
    # benchmarks.update(data['Benchmarks'])
    del data['Benchmarks']
    del data['annotations_from_paper']['Benchmarks']
    json.dump(data, open(file, 'w'), indent=4)
sorted_benchmarks = sorted(benchmarks)
print(len(sorted_benchmarks))
print(sorted_benchmarks)
print(max_bench)
