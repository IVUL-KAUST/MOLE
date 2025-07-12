from pydantic import BaseModel, ConfigDict
from pydantic import model_validator
import json
import random
from type_classes import *
random.seed(42)
ANSWER_MAX = 1000



units = ['tokens', 'sentences', 'documents', 'images', 'videos', 'hours']
dialects = ["Classical Arabic","Modern Standard Arabic","United Arab Emirates","Bahrain","Djibouti","Algeria","Egypt","Iraq","Jordan","Comoros","Kuwait","Lebanon","Libya","Morocco","Mauritania","Oman","Palestine","Qatar","Saudi Arabia","Sudan","Somalia","South Sudan","Syria","Tunisia","Yemen","Levant","North Africa","Gulf","mixed"]
languages = ['Arabic', 'English', 'French', 'Spanish', 'German', 'Greek', 'Bulgarian', 'Russian', 'Turkish', 'Vietnamese', 'Thai', 'Chinese', 'Simplified Chinese', 'Hindi', 'Swahili', 'Urdu', 'Bengali', 'Finnish', 'Japanese', 'Korean', 'Telugu', 'Indonesian', 'Italian', 'Polish', 'Portuguese', 'Estonian', 'Haitian Creole', 'Eastern Apur\u00edmac Quechua', 'Tamil', 'Sinhala']
tasks = ["machine translation", "speech recognition", "sentiment analysis", "language modeling", "topic classification", "dialect identification", "text generation", "cross-lingual information retrieval", "named entity recognition", "question answering", "multiple choice question answering", "information retrieval", "part of speech tagging", "language identification", "summarization", "speaker identification", "transliteration", "morphological analysis", "offensive language detection", "review classification", "gender identification", "fake news detection", "dependency parsing", "irony detection", "meter classification", "natural language inference", "instruction tuning", "linguistic acceptability", "commonsense reasoning", "word prediction", "image captioning", "word similarity", "grammatical error correction", "intent classification", "sign language recognition", "optical character recognition", "fill-in-the blank", "relation extraction", "stance detection", "emotion classification", "semantic parsing", "text to SQL", "lexicon analysis", "embedding evaluation", "other"]
hosts = ['GitHub', 'CodaLab', 'data.world', 'Dropbox', 'Gdrive', 'LDC', 'MPDI', 'Mendeley Data', 'Mozilla', 'OneDrive', 'QCRI Resources', 'ResearchGate', 'sourceforge', 'zenodo', 'HuggingFace', 'ELRA', 'other']
domains = ['social media', 'news articles', 'reviews', 'commentary', 'books', 'wikipedia', 'web pages', 'public datasets', 'TV Channels', 'captions', 'LLM', 'other']
collection_styles = ['crawling', 'human annotation', 'machine annotation', 'manual curation', 'LLM generated', 'other']
licenses = ['Apache-1.0', 'Apache-2.0', 'Non Commercial Use - ELRA END USER', 'BSD', 'CC BY 1.0', 'CC BY 2.0', 'CC BY 3.0', 'CC BY 4.0', 'CC BY-NC 1.0', 'CC BY-NC 2.0', 'CC BY-NC 3.0', 'CC BY-NC 4.0', 'CC BY-NC-ND 1.0', 'CC BY-NC-ND 2.0', 'CC BY-NC-ND 3.0', 'CC BY-NC-ND 4.0', 'CC BY-SA 1.0', 'CC BY-SA 2.0', 'CC BY-SA 3.0', 'CC BY-SA 4.0', 'CC BY-NC 1.0', 'CC BY-NC 2.0', 'CC BY-NC 3.0', "CC BY-NC-SA 1.0","CC BY-NC-SA 2.0","CC BY-NC-SA 3.0","CC BY-NC-SA 4.0", 'CC BY-NC 4.0', 'CC0', 'CDLA-Permissive-1.0', 'CDLA-Permissive-2.0', 'GPL-1.0', 'GPL-2.0', 'GPL-3.0', 'LDC User Agreement', 'LGPL-2.0', 'LGPL-3.0', 'MIT License', 'ODbl-1.0', 'MPL-1.0', 'MPL-2.0', 'ODC-By', 'AFL-3.0', 'CDLA-SHARING-1.0', 'unknown', 'custom']
form = ['text', 'audio', 'images', 'videos'] 
ethical_risks = ['Low', 'Medium', 'High']
access = ['Free', 'Upon-Request', 'With-Fee']
venue_types = ['preprint', 'workshop', 'conference', 'journal']


class MainSchema(BaseModel):
    @model_validator(mode='before') # validate based on the type of the field
    def validate_a(cls, data):
        for key, value in cls.model_fields.items():
            metadata = value.metadata[0]      
            data[key] = metadata.get_default() if data[key] is None else data[key]
        
        return data

class Subset(MainSchema):
    Name: Field(Str, 1, 5) # type: ignore
    Volume: Field(Float, 0) # type: ignore
    Unit: Field(Str, 1, 1, units) # type: ignore

class ArSubset(Subset):
    Dialect: Field(Str, 1, 1, dialects) # type: ignore

class MultiSubset(Subset):
    Language: Field(Str, 2, 30, languages) # type: ignore

class BaseSchema(Subset):
    model_config = ConfigDict(extra='forbid', strict=False)
    License: Field(Str, 1, 1, licenses) # type: ignore
    Link: Field(URL, 0, 1) # type: ignore
    HF_Link: Field(URL, 0, 1) # type: ignore
    Year: Field(Year, 1900, 2025) # type: ignore
    Domain: Field(List[Str], 1, len(domains), domains) # type: ignore
    Form: Field(Str, 1, 1, form) # type: ignore
    Collection_Style: Field(List[Str], 1, len(collection_styles), collection_styles) # type: ignore
    Description: Field(Str, 0, 50) # type: ignore
    Ethical_Risks: Field(Str, 1, 1, ethical_risks) # type: ignore
    Provider: Field(List[Str], 0) # type: ignore
    Derived_From: Field(List[Str], 0) # type: ignore
    Paper_Title: Field(Str, 1) # type: ignore
    Paper_Link: Field(URL, 1, 1) # type: ignore
    Tokenized: Field(Bool, 1, 1) # type: ignore
    Host: Field(Str, 1, 1, hosts) # type: ignore
    Access: Field(Str, 1, 1, access) # type: ignore
    Cost: Field(Str, 0, 1) # type: ignore
    Test_Split: Field(Bool, 1, 1) # type: ignore
    Tasks: Field(List[Str], 1, 5, tasks) # type: ignore
    Venue_Title: Field(Str, 1) # type: ignore
    Venue_Type: Field(Str, 1, 1, venue_types) # type: ignore
    Venue_Name: Field(Str, 0) # type: ignore
    Authors: Field(List[Str], 0) # type: ignore
    Affiliations: Field(List[Str], 0) # type: ignore
    Abstract: Field(Str, 1) # type: ignore


class ArSchema(BaseSchema):
    Subsets: Field(List[ArSubset], 0, len(dialects)) # type: ignore
    Dialect: Field(Str, 1, 1, dialects) # type: ignore
    Language: Field(Str, 1, 1, ['ar', 'multilingual']) # type: ignore
    Script: Field(Str, 1, 1, ['Arab', 'Latin', 'Arab-Latin']) # type: ignore

class EnSchema(BaseSchema):
    Language: Field(Str, 1, 1, ['en', 'multilingual']) # type: ignore

class JpSchema(BaseSchema):
    Language: Field(Str, 1, 1, ['jp', 'multilingual']) # type: ignore
    Script: Field(Str, 1, 1, ['Hiragana', 'Katakana', 'Kanji', 'mixed']) # type: ignore

class RuSchema(BaseSchema):
    Language: Field(Str, 1, 1, ['ru', 'multilingual']) # type: ignore

class FrSchema(BaseSchema):
    Language: Field(Str, 1, 1, ['fr', 'multilingual']) # type: ignore

class MultiSchema(BaseSchema):
    Subsets: Field(List[MultiSubset], 0, len(languages)) # type: ignore 
    Language: Field(List[Str], 2, len(languages), languages) # type: ignore

class Sons(MainSchema):
    Name: Field(Str, 1, 1) # type: ignore
    Age: Field(Int, 1, 100) # type: ignore

class TestSchema(MainSchema):
    Name: Field(Str, 1, 5) # type: ignore   
    Age: Field(Int, 1, 100) # type: ignore
    Website: Field(URL, 1, 1) # type: ignore
    Hobbies: Field(List[Str], 1, 4, ['reading', 'swimming', 'coding', 'other']) # type: ignore
    Sons: Field(List[Sons], 0, 3) # type: ignore
    Married: Field(Bool, 1, 1) # type: ignore


def evaluate_metadata(gold_metadata, predicted_metadata, schema_name = 'ar', return_metrics_only = False):
    schema = Schema(schema_name)
    results = {}
    for key in gold_metadata.keys():
        if key in ['annotations_from_paper']:
            continue
        try:
            results[key] = int(schema.match_attributes(key, gold_metadata[key], predicted_metadata[key]))
        except:
            print(key, gold_metadata[key], predicted_metadata[key])
            raise ValueError(f"Invalid type: {type(gold_metadata[key])}")
    annotations_from_paper = gold_metadata['annotations_from_paper']
    annotated_attributes = [key for key in gold_metadata.keys() if key in annotations_from_paper and annotations_from_paper[key]]
    precision = sum(results.values()) / len(results)
    recall = sum([value for key, value in results.items() if key in annotated_attributes]) / len(annotated_attributes)
    f1 = 2 * precision * recall / (precision + recall)
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    results['length'] = schema.evaluate_length(predicted_metadata)
    if return_metrics_only:
        return {'precision': precision, 'recall': recall, 'f1': f1, 'length': results['length']}
    return results

def remove_spaces_keys(metadata):
    new_metadata = {}
    for key in metadata.keys():
        new_metadata[key.replace(' ', '_')] = metadata[key]
    return new_metadata

class Schema:
    def __init__(self, schema_name = "ar"):
        self.schema_name = schema_name
        self.schema = get_schema(schema_name)
        self.columns = list(self.schema.model_fields.keys())

    def validate(self, metadata):
        return json.loads(self.schema.model_validate(metadata).model_dump_json())
    
    def evaluate(self, gold_metadata, predicted_metadata):
        return evaluate_metadata(gold_metadata, predicted_metadata)
    
    def json(self):
        schema_json = {}
        for key in self.columns:
            values = {}
            ob = self.schema.model_fields[key].metadata[0]
            values['answer_type'] = ob.get_type()
            for constrain in ['answer_min', 'answer_max', 'options']:
                attr =  getattr(ob, constrain)
                if attr is not None:
                    values[constrain] = attr
            schema_json[key] = values
            
        return json.dumps(schema_json, indent=4)
    
    def dict(self):
        return json.loads(self.json())
    
    def get_options(self, key):
        schema = self.dict()
        if 'options' in schema[key]:
            return schema[key]['options']
        else:
            return None
    
    def get_answer_min(self, key):
        schema = self.dict()
        return schema[key]['answer_min']
    
    def get_answer_max(self, key):
        schema = self.dict()
        return schema[key]['answer_max']
    
    def get_system_prompt(self):
        return f"""
        You are a professional metadata extractor of datasets from research papers. 
        You will be provided 'Paper Text', 'Schema Name', 'Input Schema' and you must respond with an 'Output JSON'.
        The 'Output JSON' is a JSON with key:answer where the answer retrieves an attribute of the 'Input Schema' from the 'Paper Text'. 
        Each attribute in the 'Input Schema' has the following fields:
        'options' : If the attribute has 'options' then the answer must be at least one of the options.
        'answer_type': The output type represents the type of the answer.
        'answer_min' : The minimum length of the answer depending on the 'answer_type'.
        'answer_max' : The maximum length of the answer depending on the 'answer_type'.
        The 'Output JSON' is a JSON that can be parsed using Python `json.load()`. USE double quotes "" not single quotes '' for the keys and values.
        The 'Output JSON' must have ONLY the keys in the 'Input Schema'.
        Use the following guidlines to extract the answer from the 'Paper Text':
        {open('GUIDELINES.md').read()}
        """

    def get_answer_type(self, key):
        return self.schema.model_fields[key].annotation
    
    def get_answer_object(self, key):
        object = self.schema.model_fields[key].metadata[0]
        return object
    
    def get_default_schema(self):
        metadata = {}
        for key in self.columns:
            metadata[key] = None
        schema = self.schema.model_validate(metadata)
        return schema.model_dump()
    
    def get_default(self, key):
        type = self.get_answer_object(key)
        return type.get_default()
    
    def evaluate_length(self, metadata):
        accuracy = 0
        for key, value in self.schema.model_fields.items():
            type  = self.get_answer_type(key)
            if type == List:
                if len(metadata[key]) >= self.get_answer_min(key) and len(metadata[key]) <= self.get_answer_max(key):
                    accuracy += 1
                else:
                    print(key,metadata[key], len(metadata[key]), self.get_answer_min(key), self.get_answer_max(key))
                    raise()
            elif type == Str:
                length_metric = len(metadata[key].split(' ')) 
                if self.get_options(key) or length_metric >= self.get_answer_min(key) and length_metric <= self.get_answer_max(key):
                    accuracy += 1
                else:
                    print(key,metadata[key], length_metric, self.get_answer_min(key), self.get_answer_max(key))
                    raise()
            else:
                accuracy += 1
        return accuracy / len(self.columns)

    def match_attributes(self, key, attr1, attr2):
        t = self.get_answer_type(key)
        if t in PRIMITIVE_TYPES:
            return attr1 == attr2
        else:
            return t.compare(attr1, attr2)
    
    def fill_missing(self, metadata):
        print('Warning: fill_missing is not implemented for schema', self.schema_name)
        return metadata
    
    def get_random(self, key):
        object = self.get_answer_object(key)
        return object.get_random()
    
    def generate_metadata(self, method = 'random'):
        metadata = {}
        for key, value in self.schema.model_fields.items():
            if method == 'random':
                metadata[key] = self.get_random(key)
            elif method == 'default':
                metadata[key] = self.get_default(key)
            else:
                raise ValueError(f"Invalid method: {method}")
        return metadata

def get_schema(schema_name):
    if schema_name == 'ar':
        return ArSchema
    elif schema_name == 'en':
        return EnSchema
    elif schema_name == 'jp':
        return JpSchema
    elif schema_name == 'ru':
        return RuSchema
    elif schema_name == 'fr':
        return FrSchema
    elif schema_name == 'multi':
        return MultiSchema
    elif schema_name == 'test':
        return TestSchema
    else:
        raise ValueError(f"Invalid schema name: {schema_name}")

def validate_metadata(metadata = None, path = None, schema_name = 'ar'):
    if metadata is None and path is None:
        raise ValueError('Either metadata or path must be provided')
    if metadata is None:
        metadata = json.load(open(path))
    annotations_from_paper = None
    if 'annotations_from_paper' in metadata:
        annotations_from_paper = metadata['annotations_from_paper']
        del metadata['annotations_from_paper']
    schema = Schema(schema_name)
    # assert len(schema.columns) == 32
    results = schema.validate(metadata)
    if annotations_from_paper is not None:
        results['annotations_from_paper'] = remove_spaces_keys(annotations_from_paper)
        
    return results