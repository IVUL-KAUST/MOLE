from glob import glob 
import json 
for file in glob("testset/*.json"):
    with open(file, "r") as f:
        data = json.load(f)
    data["Volume"] = float(str(data["Volume"]).replace(",", ""))
    data["Citations"] = int(0 if data["Citations"] == "" else float(data["Citations"]))
    data["Other Tasks"] = []
    with open(file, "w") as f:
        json.dump(data, f, indent=4)
