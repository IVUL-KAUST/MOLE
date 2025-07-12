import json
from schema import Schema
from schema import validate_metadata, evaluate_metadata

# schema = Schema(schema_name = 'ar')
# print(schema.json())
# raise()

gold_metadata = {
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
schema = Schema(schema_name = 'test')
print(schema.json())

validated_metadata = validate_metadata(path = 'testfiles/test1.json', schema_name = 'test')
evaluation_results = evaluate_metadata(gold_metadata, validated_metadata, schema_name = 'test')

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test1')


validated_metadata = validate_metadata(path = 'testfiles/test2.json', schema_name = 'test')
evaluation_results = evaluate_metadata(gold_metadata, validated_metadata, schema_name = 'test', return_metrics_only=True)

for m in evaluation_results:
    if m in ['precision', 'recall', 'f1']:
        assert abs(evaluation_results[m] - 0.83) < 0.01, f'❌ {m} value should be 0.83 but got {evaluation_results[m]}'
    else:
        assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'

print('✅ passed test2')

validated_metadata = validate_metadata(path = 'testfiles/test3.json', schema_name = 'test')
assert validated_metadata['Age'] == 0, '❌ Age should be 0 but got {validated_metadata["Age"]}'
print('✅ passed test3')

validated_metadata = validate_metadata(path = 'testfiles/test4.json', schema_name = 'test')
evaluation_results = evaluate_metadata(gold_metadata, validated_metadata, schema_name = 'test', return_metrics_only=True)
assert abs(evaluation_results['length'] - 1.0) < 0.01, f'❌ length should be 1.0 but got {evaluation_results["length"]}'
print('✅ passed test4')

validated_metadata = validate_metadata(path = 'testfiles/test5.json', schema_name = 'test')
evaluation_results = evaluate_metadata(gold_metadata, validated_metadata, schema_name = 'test')

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test5')


gold_metadata = {
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


validated_metadata = validate_metadata(path = 'testfiles/test6.json', schema_name = 'test')
evaluation_results = evaluate_metadata(gold_metadata, validated_metadata, schema_name = 'test')

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test6')


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
predicted_metadata = schema.generate_metadata(method = 'default')
evaluation_results = evaluate_metadata(default_metadata, predicted_metadata, schema_name = 'test', return_metrics_only=True)
for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test7')