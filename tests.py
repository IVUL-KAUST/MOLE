import json
from utils import validate, fix_options, cast, fill_missing, evaluate_metadata, postprocess, fetch_repository_metadata, evaluate_lengths
from constants import *
from pages.search import get_metadatav2


# no mistakes
with open('testfiles/test1.json', 'r') as f:
    metadata = json.load(f)

results = validate(metadata, use_split='valid', link = 'https://arxiv.org/abs/2402.03177')

for m in results:
    assert results[m] == 1, f'❌ {m} value should be 1 but got {results[m]}'
print('✅ passed test1 [validation 1]')

# one mistake in the volume column
with open('testfiles/test2.json', 'r') as f:
    metadata = json.load(f)

results = validate(metadata, use_split='valid', link = 'https://arxiv.org/abs/2402.03177')
evaluation_subsets = schemata['ar']['evaluation_subsets']
NUM_VALIDATION_COLUMNS = len(schemata['ar']['validation_columns'])

for m in results:
    if m == 'CONTENT':
        assert results[m] == 1- (1/len(evaluation_subsets['CONTENT'])), f'❌ {m} value should be {1- (1/len(evaluation_subsets["CONTENT"]))} but got {results[m]}'
    elif m == 'AVERAGE':
        assert results[m] == 1- (1/NUM_VALIDATION_COLUMNS), f'❌ {m} value should be {1- (1/NUM_VALIDATION_COLUMNS)} but got {results[m]}'
    else:
        assert results[m] == 1, f'❌ {m} value should be 1 but got {results[m]}'
print('✅ passed test2 [validation 2]')

# overlapping tasks
with open('testfiles/test3.json', 'r') as f:
    metadata = json.load(f)

results = validate(metadata, use_split='valid', link = 'https://arxiv.org/abs/2402.03177')

for m in results:
    assert results[m] == 1, f'❌ {m} value should be 1 but got {results[m]}'
print('✅ passed test3 [validation 3]')

# fix options
with open('testfiles/test4.json', 'r') as f:
    metadata = json.load(f)
new_metadata = fix_options(metadata)
for c in metadata:
    if c == 'Dialect':
        assert new_metadata['Dialect'] == 'Modern Standard Arabic', '❌ Modern Standard Arabic != ' + new_metadata['Dialect']
    elif c == 'Collection Style':
        assert new_metadata['Collection Style'] == ['crawling', 'LLM generated', 'manual curation'], f"❌ ['crawling', 'LLM generated', 'manual curation'] != {new_metadata['Collection Style']}"
    else:
        assert new_metadata[c] == metadata[c], f'❌ {c} should be {metadata[c]} but got {new_metadata[c]}'
print('✅ passed test4 [fix options]')

# cast
with open('testfiles/test5.json', 'r') as f:
    metadata = json.load(f)

new_metadata = cast(metadata)

for c in metadata:
    if c == 'Year':
        assert new_metadata['Year'] == 2021, "❌ 2021 != " + new_metadata['Year']
    else:
        assert new_metadata[c] == metadata[c], f'❌ {c} should be {metadata[c]} but got {new_metadata[c]}'

print('✅ passed test5 [casting]')

# fill missing
with open('testfiles/test6.json', 'r') as f:
    metadata = json.load(f)

new_metadata = fill_missing(metadata)
columns = schemata['ar']['columns']
for c in columns:
    assert c in new_metadata, f'❌ {c} should be in the metadata but it is not'

print('✅ passed test6 [fill missing]')

# with open('testfiles/example.tex', 'r') as f:
#     paper_text = f.read()

# msg, pred_metadata, cost =  get_metadatav2(paper_text, model_name = 'google/gemini-flash-1.5')
# pred_metadata = postprocess(pred_metadata)

# with open('testfiles/test7.json', 'r') as f:
#     gold_metadata = json.load(f)
# results = evaluate_metadata(pred_metadata, gold_metadata)
# print(results['AVERAGE'] == 1, f'❌ AVERAGE value should be 1 but got {results["AVERAGE"]}')
# print('✅ passed test7 [extract metadata]')

# with open('testfiles/example2.tex', 'r') as f:
#     paper_text = f.read()

# _,pred_metadata, cost =  get_metadatav2(paper_text, model_name = 'google/gemini-flash-1.5')
# readme = fetch_repository_metadata(pred_metadata['Link'])
# _,pred_metadata, cost =  get_metadatav2(metadata = pred_metadata,  model_name = 'google/gemini-flash-1.5', readme = readme)
# pred_metadata = postprocess(pred_metadata)

# with open('testfiles/test8.json', 'r') as f:
#     gold_metadata = json.load(f)
# results = evaluate_metadata(pred_metadata, gold_metadata)
# print(results['AVERAGE'] == 1, f'❌ AVERAGE value should be 1 but got {results["AVERAGE"]}')
# print('✅ passed test8 [browsing]')

pred_metadata = postprocess({})

with open('testfiles/test9.json', 'r') as f:
    gold_metadata = json.load(f)
results = evaluate_metadata(pred_metadata, gold_metadata)
assert results['AVERAGE'] == 1, f'❌ AVERAGE value should be 1 but got {results["AVERAGE"]}'
print('✅ passed test9 [empty metadata]')

# different tasks
with open('testfiles/test10.json', 'r') as f:
    metadata = json.load(f)

results = validate(metadata, use_split='valid', link = 'https://arxiv.org/abs/2402.03177')

for m in results:
    assert results[m] == 1, f'❌ {m} value should be 1 but got {results[m]}'
print('✅ passed test10 [different tasks]')

with open('testfiles/test11.json', 'r') as f:
    metadata = json.load(f)

results = evaluate_lengths(metadata, schema = "ar")

assert results == 0.96875, f'❌ results should be 0.96875 but got {results}'
print('✅ passed test11 [evaluate lengths]')

