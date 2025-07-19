#type: ignore
import json
from openai import OpenAI
from src.schema import Schema, Field, Str, Int, URL, Bool, List, ArSchema, RuSchema
from pages.search import extract_paper_text
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

class Person(Schema):
    Name: Field(Str, 1, 1)
    Age: Field(Int, 1, 100)

class Parent(Person):
    Website: Field(URL, 1, 1)
    Hobbies: Field(List[Str], 1, 4, options = ['reading', 'swimming', 'coding'])
    Married: Field(Bool, 1, 1)
    Sons: Field(List[Person], 0, 3)

template = Parent.schema_to_template()
print(template)

chat_response = client.chat.completions.create(
    model="numind/NuExtract-2.0-8B",
    temperature=0,
    messages=[
        {
            "role": "user", 
            "content": [{"type": "text", "text": "My name is Ahmad and I am 20 years old. I am interested in reading. I am married and I have a 4 year old son named Ali. My website is https://www.google.com"}],
        },
    ],
    extra_body={
        "chat_template_kwargs": {
            "template": json.dumps(json.loads(template), indent=4)
        },
    }
)
predicted_metadata = json.loads(chat_response.choices[0].message.content)


gold_metadata  = {
    "Name": "Ahmad",
    "Age": 20,
    "Website": "https://www.google.com",
    "Hobbies": ["reading"],
    'Married': True,
    "Sons": [
        {
            "Name": "Ali",
            "Age": 4
        }
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
predicted_metadata = Parent(metadata=predicted_metadata)
evaluation_results = predicted_metadata.compare_with(gold_metadata)
print(evaluation_results)

template = RuSchema.schema_to_template()

# load pdf from https://arxiv.org/pdf/2402.03177
pdf_path = "static/papers/2210.12814"
pdf_text = extract_paper_text(pdf_path, use_pdf=True)

chat_response = client.chat.completions.create(
    model="numind/NuExtract-2.0-8B",
    temperature=0,
    messages=[
        {
            "role": "user", 
            "content": [{"type": "text", "text": pdf_text}],
        },
    ],
    extra_body={
        "chat_template_kwargs": {
            "template": json.dumps(json.loads(template), indent=4)
        },
    }
)
predicted_metadata = json.loads(chat_response.choices[0].message.content)

predicted_metadata = RuSchema(metadata=predicted_metadata)

gold_metadata = json.load(open('/ibex/ai/home/alyafez/MOLE/evals/ru/test/rucola.json'))
evaluation_results = predicted_metadata.compare_with(gold_metadata)
print(evaluation_results)