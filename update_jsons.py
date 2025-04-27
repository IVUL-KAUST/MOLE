import json

def get_base_schema(has_subsets = False):
    with open("schema/ar.json", 'r') as f:
        base_schema = json.load(f)

    del base_schema['Dialect']
    if not has_subsets:
        del base_schema['Subsets']
    del base_schema["Language"]["option_description"]["ar"]
    del base_schema["Script"]
    return base_schema

langs = ['en', 'jp', 'ru', 'fr', 'multi']
full_names = {
    'en': 'English',
    'jp': 'Japanese',
    'fr': 'French',
    'ru': 'russian'
}
for lang in langs:
    base_schema = get_base_schema(has_subsets= (lang == 'multi'))
    if lang == 'multi':
        base_schema["Language"]["options"] = ["Arabic", "English", "French", "Spanish", "German", "Greek", "Bulgarian", "Russian", "Turkish", "Vietnamese", "Thai", "Chinese", "Simplified Chinese", "Hindi", "Swahili", "Urdu", "Bengali", "Finnish", "Japanese", "Korean", "Telugu", "Indonesian", "Italian", "Polish", "Portuguese"]
        base_schema["Language"]["answer_type"] = "List[str]"
        base_schema["Language"]["answer_min"] = 2
        base_schema["Language"]["answer_max"] = 25
        base_schema["Subsets"]["answer_type"] = base_schema["Subsets"]["answer_type"].replace('Dialect', 'Language')
        base_schema["Subsets"]["question"] = base_schema["Subsets"]["question"].replace('Dialect', 'Language').replace('dialect', 'language')
        del base_schema["Language"]["option_description"]
        
    else:
        base_schema["Language"]["options"][0] = lang
        desc = base_schema["Language"]["option_description"]
        full_name = full_names[lang]
        base_schema["Language"]["option_description"][lang] = f"the dataset is purely in {full_name}, there are no other languages involved"
        base_schema["Volume"]["question"] = f"What is the size of the dataset?. If the dataset is multilingual only use the size of the {full_name} dataset"
    with open(f'schema/{lang}.json', 'w') as f:
        json.dump(base_schema, f, indent=4)
