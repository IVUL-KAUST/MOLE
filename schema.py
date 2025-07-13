# type: ignore

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

class Schema(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=False)
    def __init__(self, path = None, metadata = None):
        if path is not None:
            metadata = json.load(open(path))
        elif metadata is not None:
            metadata = metadata
        else:
            raise ValueError('Either path or metadata must be provided')
        super().__init__(**metadata)

    @classmethod
    def get_attributes(cls):
        return [key for key in cls.model_fields.keys() if key not in ['annotations_from_paper']]

    @classmethod
    def schema(cls):    
        schema_json = {}
        for key in cls.get_attributes():
            values = {}
            ob = cls.model_fields[key].metadata[0]
            values['answer_type'] = ob.get_type()
            for constrain in ['answer_min', 'answer_max', 'options']:
                attr =  getattr(ob, constrain)
                if attr is not None:
                    values[constrain] = attr
            schema_json[key] = values
            
        return json.dumps(schema_json, indent=4)
    
    @classmethod
    def dict(cls):
        return json.loads(cls.schema())
    
    def json(self):
        return json.loads(self.model_dump_json())
    
    @classmethod
    def get_options(cls, key):
        schema = cls.dict()
        if 'options' in schema[key]:
            return schema[key]['options']
        else:
            return None
    
    @classmethod
    def get_answer_min(cls, key):
        schema = cls.dict()
        return schema[key]['answer_min']
    
    @classmethod
    def get_answer_max(cls, key):
        schema = cls.dict()
        return schema[key]['answer_max']
    
    def get_system_prompt():
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
        return self.model_fields[key].annotation
    

    @classmethod
    def get_answer_object(cls, key):
        object = cls.model_fields[key].metadata[0]
        return object
    
    @classmethod
    def get_default_schema(cls):
        metadata = {}
        for key in cls.get_attributes():
            metadata[key] = None
        schema = cls.model_validate(metadata)
        return schema.model_dump()
    
    @classmethod
    def get_default(cls, key):
        type = cls.get_answer_object(key)
        return type.get_default()
    
    def evaluate_length(self):
        accuracy = 0
        metadata = self.model_dump()
        for key in self.get_attributes():
            type  = self.get_answer_object(key)
            length = type.validate_length(metadata[key])
            accuracy += length
        return accuracy / len(self.get_attributes())
    
    def compare_with(self, gold_metadata, return_metrics_only = False):
        results = {}
        for key in gold_metadata.keys():
            if key in ['annotations_from_paper']:
                continue
            try:
                results[key] = int(self.match_attributes(key, gold_metadata[key], self.model_dump()[key]))
            except:
                print(key, gold_metadata[key], self.model_dump()[key])
                raise ValueError(f"Invalid type: {type(gold_metadata[key])}")
        annotations_from_paper = gold_metadata['annotations_from_paper']
        annotated_attributes = [key for key in gold_metadata.keys() if key in annotations_from_paper and annotations_from_paper[key]]
        precision = sum(results.values()) / len(results)
        recall = sum([value for key, value in results.items() if key in annotated_attributes]) / len(annotated_attributes)
        f1 = 2 * precision * recall / (precision + recall)
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = f1
        if return_metrics_only:
            return {'precision': precision, 'recall': recall, 'f1': f1, 'length': self.evaluate_length()}
        return results

    def match_attributes(self, key, attr1, attr2):
        t = self.get_answer_object(key)   
        return t.compare(attr1, attr2)
    
    @classmethod
    def get_random(cls, key):
        object = cls.get_answer_object(key)
        return object.get_random()
    
    @classmethod
    def generate_metadata(cls, method = 'random'):
        metadata = {}
        for key in cls.get_attributes():
            if method == 'random':
                metadata[key] = cls.get_random(key)
            elif method == 'default':
                metadata[key] = cls.get_default(key)
            else:
                raise ValueError(f"Invalid method: {method}")
        return cls(metadata = metadata)

    
    @model_validator(mode='before') # validate based on the type of the field
    def validate_a(cls, data):
        for key, value in cls.model_fields.items():
            type = value.metadata[0]
            data[key] = type.get_default() if data[key] is None else data[key]
        
        return data
       

class Subset(Schema):
    Name: Field(Str, 1, 5)
    Volume: Field(Float, 0)
    Unit: Field(Str, 1, 1, units)

class ArSubset(Subset):
    Dialect: Field(Str, 1, 1, dialects)

class MultiSubset(Subset):
    Language: Field(Str, 2, 30, languages)

class DatasetSchema(Subset):
    model_config = ConfigDict(extra='forbid', strict=False)
    License: Field(Str, 1, 1, licenses)
    Link: Field(URL, 0, 1)
    HF_Link: Field(URL, 0, 1)
    Year: Field(Year, 1900, 2025)
    Domain: Field(List[Str], 1, len(domains), domains)
    Form: Field(Str, 1, 1, form)
    Collection_Style: Field(List[Str], 1, len(collection_styles), collection_styles)
    Description: Field(Str, 0, 50)
    Ethical_Risks: Field(Str, 1, 1, ethical_risks)
    Provider: Field(List[Str], 0, 10)
    Derived_From: Field(List[Str], 0, 10)
    Paper_Title: Field(Str, 1, 100)
    Paper_Link: Field(URL, 1, 1)
    Tokenized: Field(Bool, 1, 1)
    Host: Field(Str, 1, 1, hosts)
    Access: Field(Str, 1, 1, access)
    Cost: Field(Str, 0, 1)
    Test_Split: Field(Bool, 1, 1)
    Tasks: Field(List[Str], 1, 5, tasks)
    Venue_Title: Field(Str, 1, 1)
    Venue_Type: Field(Str, 1, 1, venue_types)
    Venue_Name: Field(Str, 0, 10)
    Authors: Field(List[Str], 0, 100)
    Affiliations: Field(List[Str], 0, 100)
    Abstract: Field(Str, 1, 1000)


class ArSchema(DatasetSchema):
    Subsets: Field(List[ArSubset], 0, len(dialects))
    Dialect: Field(Str, 1, 1, dialects)
    Language: Field(Str, 1, 1, ['ar', 'multilingual'])
    Script: Field(Str, 1, 1, ['Arab', 'Latin', 'Arab-Latin'])

class EnSchema(DatasetSchema):
    Language: Field(Str, 1, 1, ['en', 'multilingual'])

class JpSchema(DatasetSchema):
    Language: Field(Str, 1, 1, ['jp', 'multilingual'])
    Script: Field(Str, 1, 1, ['Hiragana', 'Katakana', 'Kanji', 'mixed'])

class RuSchema(DatasetSchema):
    Language: Field(Str, 1, 1, ['ru', 'multilingual'])

class FrSchema(DatasetSchema):
    Language: Field(Str, 1, 1, ['fr', 'multilingual'])

class MultiSchema(DatasetSchema):
    Subsets: Field(List[MultiSubset], 0, len(languages))
    Language: Field(List[Str], 2, len(languages), languages)


def remove_spaces_keys(metadata):
    new_metadata = {}
    for key in metadata.keys():
        new_metadata[key.replace(' ', '_')] = metadata[key]
    return new_metadata


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