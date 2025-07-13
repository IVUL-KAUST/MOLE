# type: ignore

from pydantic import BaseModel, GetCoreSchemaHandler, model_validator
from pydantic_core import CoreSchema
from typing import Any, Annotated
from dataclasses import dataclass
import random
import string

MAX_INT = 1000000000000000000
@dataclass(frozen=True)
class Constraints:
    answer_min: int
    answer_max: int = MAX_INT
    pattern: str = None
    options: list[str] = None

def Field(field_type, answer_min=0, answer_max=MAX_INT, options=None):
    return Annotated[field_type, field_type(answer_min=answer_min, answer_max=answer_max, options=options)]

class BaseType:
    base_type = None
    def __init__(self, answer_min=0, answer_max=MAX_INT, options=None, field_names=None):
        self.answer_min = answer_min
        self.answer_max = answer_max
        self.options = options
        self.field_names = field_names
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        # Return the handler for int directly, don't wrap in cls
        return handler(cls.base_type)
    
    @classmethod
    def compare(cls, attr1, attr2):
        return attr1 == attr2

class Float(BaseType):
    base_type = float

    def get_random(self, r = True):
        value = random.uniform(self.answer_min, self.answer_max)
        if r:
            return round(value, 2)
        else:
            return value
    
    def get_default(self):
        return 0.0
    
    def get_type(self):
        return 'float'
    
    def cast(self, value):
        return float(value)
    
    def validate_length(self, value):
        if value >= self.answer_min and value <= self.answer_max:
            return 1
        else:
            return 0
    
class Int(BaseType):
    base_type = int

    def get_random(self):
        return random.randint(self.answer_min, self.answer_max)
    
    def get_default(self):
        return 0
    
    def get_type(self):
        return 'int'
    
    def validate_length(self, value):
        if value >= self.answer_min and value <= self.answer_max:
            return 1
        else:
            return 0
    
class Bool(BaseType):
    base_type = bool

    def get_random(self):
        return random.choice([True, False])
    
    def get_default(self):
        return False

    def get_type(self):
        return 'bool'

    def validate_length(self, value):
        return 1
    
    
class Year(BaseType):
    base_type = int

    def get_random(self):
        return random.randint(self.answer_min, self.answer_max)
    
    def get_default(self):
        return 2025
    
    def get_type(self):
        return 'year'
    
    def validate_length(self, value):
        return value >= self.answer_min and value <= self.answer_max
    
class URL(BaseType):
    base_type = str

    def get_random(self):
        return 'https://www.{}.com'.format(random.choice(string.ascii_letters))

    def get_default(self):
        return ''
    
    def get_type(self):
        return 'url'
    
    def validate_length(self, value):
        metric = value.split(' ')
        return len(metric) >= self.answer_min and len(metric) <= self.answer_max
    
class Str(BaseType):
    base_type = str

    def get_random(self):
        if self.options:
            return random.choice(self.options)
        else:
            # answer_min is the number of words 
            return ' '.join(random.choices(string.ascii_letters, k=random.randint(self.answer_min, self.answer_max)))
    
    def get_default(self):
        return ''
    
    def get_type(self):
        return 'str'
    
    def validate_length(self, value):
        metric = value.split(' ')
        return len(metric) >= self.answer_min and len(metric) <= self.answer_max or self.options is not None
    
class List(BaseType):
    base_type = list
    _inner_type = None

    @classmethod
    def __class_getitem__(cls, item):
        class ListType(cls):
            _inner_type = item
        return ListType
    
    def get_random(self):
        if self.options:
            return random.sample(self.options, random.randint(self.answer_min, self.answer_max))
        else:
            return []

    def get_default(self):
        return []

    def get_type(self):
        # Check if inner type is a Pydantic model
        if hasattr(self._inner_type, 'model_fields'):
            # Extract field names from the Pydantic model
            field_names = list(self._inner_type.model_fields.keys())
            fields_str = ', '.join(field_names)
            return f'list[dict[{fields_str}]]'
        else:
            return f'list[{self._inner_type().get_type() if hasattr(self._inner_type(), "get_type") else self._inner_type}]'
    
    @classmethod
    def compare(cls, attr1, attr2):
        if len(attr1) != len(attr2):
            return False
        else:
            for item in attr1:
                if item not in attr2:
                    return False
            return True
    
    def validate_length(self, value):
        return len(value) >= self.answer_min and len(value) <= self.answer_max
    
class Cars(BaseModel):
    Model: Field(Str)
    Color: Field(Str, options=['Red', 'Blue', 'Green'])

class Person(BaseModel):
    Age: Field(Int)
    Name: Field(Str, 1, 5)
    Hobbies: Field(List[Str], 1, 2, options=['reading', 'swimming', 'coding', 'other'])
    Cars: Field(List[Cars])
    Married: Field(Bool)
    Website: Field(URL)
    Salary: Field(Float, 1000, 100000)
    
    @model_validator(mode='before') # validate based on the type of the field
    def validate_a(cls, data):
        for key, value in cls.model_fields.items():
            metadata = value.metadata[0]      
            data[key] = metadata.get_random() if data[key] is None else data[key]
            # data[key] = metadata.default if data[key] is None else data[key]
        
        return data

PRIMITIVE_TYPES = [Int, Float, Bool, Year, URL, Str]

if __name__ == '__main__':
    test = {
        'Age': None,
        'Name': None,
        'Hobbies': None,
        'Cars': [{
            'Model': 'Toyota',
            'Color': 'Red'
        }],
        'Married': None,
        'Website': None,
        'Salary': None
    }

    print(Person.model_validate(test).model_dump_json())
    for field, value in Person.model_fields.items():
        print(field, value.metadata[0].get_type())
