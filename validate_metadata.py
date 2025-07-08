from schema import get_schema, ArSchema, evaluate_metadata, validate_metadata, Schema
from pydantic import ValidationError
from glob import glob
import json
metadata = {
        "Name": "MGB-3 Arabic Challenge",
        "Volume": 16.0,
        "Unit": "hours",
        "License": "unknown",
        "Link": "http://www.mgb-challenge.org/workshop.html",
        "HF_Link": "https://github.com/qcri/dialectID",
        "Year": 2017,
        "Domain": [
            "social media"
        ],
        "Form": "spoken",
        "Collection_Style": [
            "crawling"
        ],
        "Description": "The MGB-3 Arabic Challenge focused on Egyptian dialect speech obtained from YouTube, comprising 16 hours of audio across seven genres.",
        "Ethical_Risks": "Medium",
        "Provider": [
            "Qatar Computing Research Institute",
            "University of Edinburgh"
        ],
        "Derived_From": [
            "MGB-2"
        ],
        "Paper_Title": "Speech Recognition Challenge in the Wild: Arabic MGB-3",
        "Paper_Link": "https://www.researchgate.net/publication/318499999_Speech_Recognition_Challenge_in_the_Wild_Arabic_MGB-3",
        "Tokenized": False,
        "Host": "GitHub",
        "Access": "Free",
        "Cost": "",
        "Test_Split": True,
        "Tasks": [
            "speech recognition",
            "dialect identification"
        ],
        "Venue_Title": "Interspeech",
        "Venue_Type": "conference",
        "Venue_Name": "",
        "Authors": [
            "Ahmed Ali",
            "Stephan Vogel",
            "Steve Renals"
        ],
        "Affiliations": [
            "Qatar Computing Research Institute, HBKU, Doha, Qatar",
            "Centre for Speech Technology Research, University of Edinburgh, UK"
        ],
        "Abstract": "This paper describes the Arabic MGB-3 Challenge -- Arabic Speech Recognition in the Wild. Unlike last year's Arabic MGB-2 Challenge, for which the recognition task was based on more than 1,200 hours broadcast TV news recordings from Aljazeera Arabic TV programs, MGB-3 emphasises dialectal Arabic using a multi-genre collection of Egyptian YouTube videos. Seven genres were used for the data collection: comedy, cooking, family/kids, fashion, drama, sports, and science (TEDx). A total of 16 hours of videos, split evenly across the different genres, were divided into adaptation, development and evaluation data sets. The Arabic MGB-Challenge comprised two tasks: A) Speech transcription, evaluated on the MGB-3 test set, along with the 10 hour MGB-2 test set to report progress on the MGB-2 evaluation;  B) Arabic dialect identification, introduced this year in order to distinguish between four major Arabic dialects -- Egyptian, Levantine, North African, Gulf, as well as Modern Standard Arabic. Two hours of audio per dialect were released for development and a further two hours were used for evaluation. For dialect identification, both lexical features and i-vector bottleneck features were shared with participants in addition to the raw audio recordings.  Overall, thirteen teams submitted ten systems to the challenge. We outline the approaches adopted in each system, and summarise the evaluation results.",
        "Subsets": [
            {
                "Name": "Comedy",
                "Volume": 0.6,
                "Unit": "hours",
                "Dialect": "Egyptian"
            },
            {
                "Name": "Cooking",
                "Volume": 0.6,
                "Unit": "hours",
                "Dialect": "Egyptian"
            },
            {
                "Name": "Family/Kids",
                "Volume": 0.8,
                "Unit": "hours",
                "Dialect": "Egyptian"
            },
            {
                "Name": "Fashion",
                "Volume": 0.6,
                "Unit": "hours",
                "Dialect": "Egyptian"
            },
            {
                "Name": "Drama",
                "Volume": 0.6,
                "Unit": "hours",
                "Dialect": "Egyptian"
            },
            {
                "Name": "Science",
                "Volume": 0.6,
                "Unit": "hours",
                "Dialect": "Egyptian"
            },
            {
                "Name": "Sports",
                "Volume": 0.8,
                "Unit": "hours",
                "Dialect": "Egyptian"
            }
        ],
        "Dialect": "Egypt",
        "Language": "ar",
        "Script": "Arab"
}
schema = Schema('ar')
predicted_metadata = validate_metadata(metadata = metadata, schema_name = 'ar')
gold_metadata = validate_metadata(path = 'evals/ar/test/mgb-3.json', schema_name = 'ar')
print(evaluate_metadata(gold_metadata, predicted_metadata, schema_name = 'ar'))
print()

# print(Schema('ar').json())

for schema_name in ['ar', 'en', 'jp', 'ru', 'fr', 'multi']:
    for file in glob(f'evals/{schema_name}/**/*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            del data['annotations_from_paper']
            keys = list(data.keys())
            metadata = {}
            for key in keys:
                metadata[key.replace(' ', '_')] = data[key]
            try:
                schema = get_schema(schema_name)
                schema.model_validate(metadata)
            except ValidationError as e:
                print(f"Validation error in {file}: {e}")