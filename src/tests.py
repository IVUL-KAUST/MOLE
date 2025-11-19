# type: ignore

from schema import Schema, Parent
from pydantic import Field
from type_classes import *
from search import run
from rich import print


gold_metadata1  = {
    "Name": "ahmad",
    "Age": 20,
    "Website": "https://www.google.com",
    "Hobbies": ["reading"],
    'Married': True,
    "Sons":[],
    "annotations_from_paper": {
        "Name": 1,
        "Age": 1,
        "Website": 1,
        "Hobbies": 1,
        "Sons": 1,
        "Married": 1
    }
}

predicted_metadata = Parent(
    path = 'testfiles/test1.json'
)
print(predicted_metadata.schema())
evaluation_results = predicted_metadata.compare_with(gold_metadata1)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test1')


# [reading] - > [reading, swimming]
validated_metadata = Parent(
    path = 'testfiles/test2.json'
)
evaluation_results = validated_metadata.compare_with(gold_metadata1)
assert evaluation_results['Hobbies'] == 0.5, f'❌ Hobbies value should be 0.5 but got {evaluation_results["Hobbies"]}'

print('✅ passed test2')

validated_metadata = Parent(
    path = 'testfiles/test3.json'
)
assert validated_metadata.json()['Age'] == 0, '❌ Age should be 0 but got {validated_metadata["Age"]}'
print('✅ passed test3')

validated_metadata = Parent(
    path = 'testfiles/test4.json'
)
evaluation_results = validated_metadata.compare_with(gold_metadata1, return_metrics_only=True)
assert abs(evaluation_results['length'] - 0.83) < 0.01, f'❌ length should be 0.83 but got {evaluation_results["length"]}'
print('✅ passed test4')

validated_metadata = Parent(
    path = 'testfiles/test5.json'
)
evaluation_results = validated_metadata.compare_with(gold_metadata1)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test5')


gold_metadata2 = {
    "Name": "ahmad",
    "Age": 20,
    "Website": "https://www.google.com",
    "Hobbies": ["swimming", "coding", "reading"],
    'Married': True,
    "Sons": [
            {"Name": "ahmad", "Age": 20},
            {"Name": "ali", "Age": 10}
    ],
    "annotations_from_paper": {
        "Name": 1,
        "Age": 1,
        "Website": 1,
        "Hobbies": 1,
        "Sons": 1,
        "Married": 1
    }
}


validated_metadata = Parent(
    path = 'testfiles/test6.json'
)
evaluation_results = validated_metadata.compare_with(gold_metadata2)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test6')


validated_metadata = Parent(
    path = 'testfiles/test7.json'
)
evaluation_results = validated_metadata.compare_with(gold_metadata1)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test7')

validated_metadata = Parent(
    path = 'testfiles/test8.json'
)
evaluation_results = validated_metadata.compare_with(gold_metadata1)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test8')

validated_metadata = Parent(
    path = 'testfiles/test9.json'
)
evaluation_results = validated_metadata.compare_with(gold_metadata1)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test9')

validated_metadata = Parent(
    path = 'testfiles/test10.json'
)
evaluation_results = validated_metadata.compare_with(gold_metadata1)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test10')

default_metadata = {
    "Name": "",
    "Age": 0,
    "Website": "",
    "Hobbies": [],
    "Sons":[],
    'Married': False,
    "annotations_from_paper": {
        "Name": 1,
        "Age": 1,
        "Website": 1,
        "Hobbies": 1,
        "Sons": 1,
        "Married": 1
    }
}

predicted_metadata = Parent.generate_metadata(method = 'default')
evaluation_results = predicted_metadata.compare_with(default_metadata, return_metrics_only=True)
for m in evaluation_results:
    if m == 'length':
        assert abs(evaluation_results[m] - 0.66) < 0.01, f'❌ {m} value should be 0.66 but got {evaluation_results[m]}'
    else:
        assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test11')

# from arg_utils import args

# args.model_name='moonshotai/kimi-k2'
# args.schema_name='model'
# args.backend='openrouter'
# args.results_path='synth_dataset_models'
# args.format='pdf_plumber'
# args.overwrite=True
# results = run("https://arxiv.org/pdf/2507.20534", args)
# print(results[args.model_name]['metadata'])