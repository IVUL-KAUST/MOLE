from datasets import load_dataset
data = load_dataset('Zaid/mole_preference')
print(data['train'][3])