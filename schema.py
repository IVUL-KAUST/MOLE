from typing import Annotated, Any, Union, Callable
from pydantic import BaseModel, ConfigDict, computed_field
from pydantic_core import CoreSchema, core_schema
from dataclasses import dataclass
import datetime as dt
from pydantic import GetCoreSchemaHandler, TypeAdapter, ValidationError
import re
from pydantic import field_validator, model_validator
import json
import random

ANSWER_MAX = 1000

@dataclass(frozen=True)
class Constraints:
    answer_min: int
    answer_max: int = ANSWER_MAX
    pattern: str = None
    options: list[str] = None

class year(int):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(int))

units = ['tokens', 'sentences', 'documents', 'images', 'videos', 'hours']
dialects = ["Classical Arabic","Modern Standard Arabic","United Arab Emirates","Bahrain","Djibouti","Algeria","Egypt","Iraq","Jordan","Comoros","Kuwait","Lebanon","Libya","Morocco","Mauritania","Oman","Palestine","Qatar","Saudi Arabia","Sudan","Somalia","South Sudan","Syria","Tunisia","Yemen","Levant","North Africa","Gulf","mixed"]
languages = ['Arabic', 'English', 'French', 'Spanish', 'German', 'Greek', 'Bulgarian', 'Russian', 'Turkish', 'Vietnamese', 'Thai', 'Chinese', 'Simplified Chinese', 'Hindi', 'Swahili', 'Urdu', 'Bengali', 'Finnish', 'Japanese', 'Korean', 'Telugu', 'Indonesian', 'Italian', 'Polish', 'Portuguese', 'Estonian', 'Haitian Creole', 'Eastern Apur\u00edmac Quechua', 'Tamil', 'Sinhala']
tasks = ["machine translation", "speech recognition", "sentiment analysis", "language modeling", "topic classification", "dialect identification", "text generation", "cross-lingual information retrieval", "named entity recognition", "question answering", "multiple choice question answering", "information retrieval", "part of speech tagging", "language identification", "summarization", "speaker identification", "transliteration", "morphological analysis", "offensive language detection", "review classification", "gender identification", "fake news detection", "dependency parsing", "irony detection", "meter classification", "natural language inference", "instruction tuning", "linguistic acceptability", "commonsense reasoning", "word prediction", "image captioning", "word similarity", "grammatical error correction", "intent classification", "sign language recognition", "optical character recognition", "fill-in-the blank", "relation extraction", "stance detection", "emotion classification", "semantic parsing", "text to SQL", "lexicon analysis", "embedding evaluation", "other"]
hosts = ['GitHub', 'CodaLab', 'data.world', 'Dropbox', 'Gdrive', 'LDC', 'MPDI', 'Mendeley Data', 'Mozilla', 'OneDrive', 'QCRI Resources', 'ResearchGate', 'sourceforge', 'zenodo', 'HuggingFace', 'ELRA', 'other']
domains = ['social media', 'news articles', 'reviews', 'commentary', 'books', 'wikipedia', 'web pages', 'public datasets', 'TV Channels', 'captions', 'LLM', 'other']
collection_styles = ['crawling', 'human annotation', 'machine annotation', 'manual curation', 'LLM generated', 'other']
licenses = ['Apache-1.0', 'Apache-2.0', 'Non Commercial Use - ELRA END USER', 'BSD', 'CC BY 1.0', 'CC BY 2.0', 'CC BY 3.0', 'CC BY 4.0', 'CC BY-NC 1.0', 'CC BY-NC 2.0', 'CC BY-NC 3.0', 'CC BY-NC 4.0', 'CC BY-NC-ND 1.0', 'CC BY-NC-ND 2.0', 'CC BY-NC-ND 3.0', 'CC BY-NC-ND 4.0', 'CC BY-SA 1.0', 'CC BY-SA 2.0', 'CC BY-SA 3.0', 'CC BY-SA 4.0', 'CC BY-NC 1.0', 'CC BY-NC 2.0', 'CC BY-NC 3.0', "CC BY-NC-SA 1.0","CC BY-NC-SA 2.0","CC BY-NC-SA 3.0","CC BY-NC-SA 4.0", 'CC BY-NC 4.0', 'CC0', 'CDLA-Permissive-1.0', 'CDLA-Permissive-2.0', 'GPL-1.0', 'GPL-2.0', 'GPL-3.0', 'LDC User Agreement', 'LGPL-2.0', 'LGPL-3.0', 'MIT License', 'ODbl-1.0', 'MPL-1.0', 'MPL-2.0', 'ODC-By', 'AFL-3.0', 'CDLA-SHARING-1.0', 'unknown', 'custom']
form = ['text', 'spoken', 'images', 'videos'] # maybe use audio instead of spoken
ethical_risks = ['Low', 'Medium', 'High'] # use lower case instead
access = ['Free', 'Upon-Request', 'With-Fee']
venue_types = ['preprint', 'workshop', 'conference', 'journal']

class Subset(BaseModel):
    Name: Annotated[str, Constraints(answer_min=1, answer_max=5)]
    Volume: float
    Unit: Annotated[str, Constraints(answer_min=1, answer_max=1, options=units)]

class ArSubset(Subset):
    Dialect: Annotated[str, Constraints(answer_min=1, answer_max=1, options=dialects)]

class MultiSubset(Subset):
    Language: Annotated[str, Constraints(answer_min=2, answer_max=30, options=languages)]

class BaseSchema(Subset):
    model_config = ConfigDict(extra='forbid', strict=False)
    License: Annotated[str, Constraints(answer_min=1, answer_max=15, options=licenses)]
    Link: Annotated[str, Constraints(pattern=r'^https?://.*$', answer_min=1, answer_max=1)]
    HF_Link: Annotated[str, Constraints(pattern=r'^https?://.*$', answer_min=0, answer_max=1)]
    Year: year
    Domain: Annotated[list[str], Constraints(answer_min=1, answer_max=len(domains), options=domains)]
    Form: Annotated[str, Constraints(answer_min=1, answer_max=1, options=form)]
    Collection_Style: Annotated[list[str], Constraints(answer_min=1, answer_max=len(collection_styles), options=collection_styles)]
    Description: Annotated[str, Constraints(answer_min=0, answer_max=50)]
    Ethical_Risks: Annotated[str, Constraints(answer_min=1, answer_max=1, options=ethical_risks)]
    Provider: Annotated[list[str], Constraints(answer_min=0)]
    Derived_From: Annotated[list[str], Constraints(answer_min=0)]
    Paper_Title: Annotated[str, Constraints(answer_min=1)]
    Paper_Link: Annotated[str, Constraints(pattern=r'^https?://.*$', answer_min=1, answer_max=1)]
    Tokenized: bool
    Host: Annotated[str, Constraints(answer_min=1, answer_max=1, options=hosts)]
    Access: Annotated[str, Constraints(answer_min=1, answer_max=1, options=access)]
    Cost: Annotated[str, Constraints(answer_min=0, answer_max=1)]
    Test_Split: bool
    Tasks: Annotated[list[str], Constraints(answer_min=1, answer_max=len(tasks), options = tasks)]
    Venue_Title: Annotated[str, Constraints(answer_min=1, answer_max=1)]
    Venue_Type: Annotated[str, Constraints(answer_min=1, answer_max=1, options=venue_types)]
    Venue_Name: Annotated[str, Constraints(answer_min=0)]
    Authors: Annotated[list[str], Constraints(answer_min=1)]
    Affiliations: Annotated[list[str], Constraints(answer_min=1)]
    Abstract: Annotated[str, Constraints(answer_min = 1)]

    @model_validator(mode='before') # validate based on the type of the field
    def not_null(cls, data):
        # data is the metadata from the json file
        for key, value in cls.model_fields.items(): # get the annotations from the data class 
            t = value.annotation.__name__ # original annotation type
            if t == 'list':
                if data[key] is None:
                    data[key] = []
            elif t == 'str':
                if data[key] is None:
                    data[key] = ''
            elif t == 'int':
                if data[key] is None:
                    data[key] = 0
            elif t == 'float':
                if data[key] is None:
                    data[key] = 0.0
            elif t == 'bool':
                if data[key] is None:
                    data[key] = False
            elif t == 'year':
                if data[key] is None:
                    data[key] = 2025
            else:
                raise ValueError(f"Invalid type: {t}")
        return data

    @model_validator(mode='after') # not sure what to do here
    def validate_after(self):
        return self

class ArSchema(BaseSchema):
    """ar"""
    Subsets: Annotated[list[ArSubset], Constraints(answer_min=0, answer_max=29)]
    Dialect: Annotated[str, Constraints(answer_min=1, answer_max=1, options=dialects)]
    Language: Annotated[str, Constraints(answer_min=1, answer_max=1, options=['ar', 'multilingual'])]
    Script: Annotated[str, Constraints(answer_min=1, answer_max=1, options=['Arab', 'Latin', 'Arab-Latin'])]

class EnSchema(BaseSchema):
    Language: Annotated[str, Constraints(answer_min=1, answer_max=1, options=['en', 'multilingual'])]

class JpSchema(BaseSchema):
    Language: Annotated[str, Constraints(answer_min=1, answer_max=1, options=['jp', 'multilingual'])]
    Script: Annotated[str, Constraints(answer_min=1, answer_max=1, options=['Hiragana', 'Katakana', 'Kanji', 'mixed'])]

class RuSchema(BaseSchema):
    Language: Annotated[str, Constraints(answer_min=1, answer_max=1, options=['ru', 'multilingual'])]

class FrSchema(BaseSchema):
    Language: Annotated[str, Constraints(answer_min=1, answer_max=1, options=['fr', 'multilingual'])]

class MultiSchema(BaseSchema):
    Subsets: Annotated[list[MultiSubset], Constraints(answer_min=0, answer_max=30)]
    Language: Annotated[list[str], Constraints(answer_min=2, answer_max=30, options=languages)]

def match_lists(list1, list2):
    if type(list1) in [str, int, float, year, bool]:
        if list1 == list2:
            return True
        else:
            return False
    for item in list1:
        if item not in list2:
            return False
    return True

def evaluate_metadata(gold_metadata, predicted_metadata, schema_name = 'ar', return_metrics_only = False):
    schema = Schema(schema_name)
    results = {}
    for key in gold_metadata.keys():
        if key in ['annotations_from_paper']:
            continue
        
        if match_lists(gold_metadata[key], predicted_metadata[key]):
            results[key] = 1
        else:
            results[key] = 0
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
        for key, value in self.schema.model_fields.items():
            values = {}
            for val in ['answer_type', 'answer_min', 'answer_max','options']:
                try:
                    if val == 'answer_type':
                        values[val] = value.annotation.__name__
                    else:
                        r = value.metadata[0].__getattribute__(val)
                        if r is not None and  r != []:
                            values[val] = r
                except:
                    pass
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
        
    def get_system_prompt(self):
        return f"""You are a professional research paper reader. You will be provided 'Input schema' and 'Paper Text' and you must respond with an 'Output JSON'.
        The 'Output JSON' is a JSON with key:answer where the answer represents an answer to a 'question' provided in the 'Input Schema'. 
        The 'Input Schema' has the following main fields:
        'question': A question that needs to be answered.
        'options' : If the 'question' has 'options' then the question can be answered by choosing one or more options depending on 'answer_min' and 'answer_max'
        'options_description': A description of the 'options' that might be unclear. Use the descriptions to understand the options. 
        'answer_type': The output type of the answer to the 'question'. The answer must follow the type of the answer. 
        'answer_min' : If the 'answer_type' is a List, then it defines the minimum number of list items in the answer. Otherwise it defines the minimum number of words in the answer.
        'answer_max' : If the 'answer_type' is a List, then it defines the maximum number of list items in the answer. Otherwise it defines the maximum number of words in the answer.
        The answer must be the same type as 'answer_type' and its length must be in the range ['answer_min', 'answer_max']. If 'answer_min' = 'answer_max' then the length of answer MUST be 'answer_min'. 
        The 'Output JSON' is a JSON that can be parsed using Python `json.load()`. USE double quotes "" not single quotes '' for the keys and values.
        The 'Output JSON' has ONLY the keys: '{self.columns}'. The value for each key is the answer to the 'question' that represents the same key in the 'Input Schema'."""

    def get_answer_type(self, key):
        return self.schema.model_fields[key].annotation.__name__
    
    def get_default_schema(self):
        metadata = {}
        for key in self.columns:
            metadata[key] = None
        schema = self.schema.model_validate(metadata)
        return schema.model_dump()
    
    def get_default_value(self, key):
        return self.get_default_schema()[key]
    
    def evaluate_length(self, metadata):
        accuracy = 0
        for key, value in self.schema.model_fields.items():
            if value.annotation.__name__ == 'list':
                if len(metadata[key]) >= value.metadata[0].answer_min and len(metadata[key]) <= value.metadata[0].answer_max:
                    accuracy += 1
            else:
                accuracy += 1
        return accuracy / len(self.columns)
    
    def fill_missing(self, metadata):
        print('Warning: fill_missing is not implemented for schema', self.schema_name)
        return metadata
    
    def generate_metadata(self, method = 'random'):
        metadata = {}
        for key, value in self.schema.model_fields.items():
            type = self.get_answer_type(key)
            options = self.get_options(key)
            if options is not None:

                if method == 'random':
                    metadata[key] = random.sample(options, random.randint(value.metadata[0].answer_min, value.metadata[0].answer_max)) if type == 'list' else random.choice(options)
                elif method == 'last':
                    metadata[key] = [options[-1]] if type == 'list' else options[-1]
                elif method == 'first':
                    metadata[key] = [options[0]] if type == 'list' else options[0]
                else:
                    raise ValueError(f"Invalid method: {method}")
            else:
                metadata[key] = self.get_default_value(key)
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

# example of casting data

"""
from __future__ import annotations

from pydantic import BaseModel


class UserIn(BaseModel):
    favorite_number: int | str


class UserOut(BaseModel):
    favorite_number: int


def my_api(user: UserIn) -> UserOut:
    favorite_number = user.favorite_number
    if isinstance(favorite_number, str):
        favorite_number = int(user.favorite_number.strip())

    return UserOut(favorite_number=favorite_number)


"""