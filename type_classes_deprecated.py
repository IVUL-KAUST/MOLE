from typing import Any
from dataclasses import dataclass
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
import random
import string
random.seed(42)

@dataclass(frozen=True)
class Constraints:
    answer_min: int
    answer_max: int = 1000
    pattern: str = None
    options: list[str] = None
    custom_type_class: Any = None  # Add this field

class CustomType(type):
    """Metaclass for creating custom types with default values and string representations"""
    
    def __new__(cls, name, bases, namespace, **kwargs):
        # Store the default value if provided
        if 'default' in kwargs:
            namespace['_default_value'] = kwargs['default']
        else:
            namespace['_default_value'] = None
            
        # Store the string type name if provided
        if 'type_name' in kwargs:
            namespace['_type_name'] = kwargs['type_name']
        else:
            namespace['_type_name'] = name.lower()
            
        # Store constraint-related parameters as class attributes
        if 'answer_min' in kwargs:
            namespace['_answer_min'] = kwargs['answer_min']
        if 'answer_max' in kwargs:
            namespace['_answer_max'] = kwargs['answer_max']
        if 'options' in kwargs:
            namespace['_options'] = kwargs['options']
            
        # Don't pass any kwargs to parent metaclass to avoid issues with base classes
        return super().__new__(cls, name, bases, namespace)
    
    def __str__(cls):
        """Return the string representation of the type"""
        return cls._type_name
    
    def __repr__(cls):
        return f"<type '{cls._type_name}'>"
    
    @property
    def default(cls):
        """Get the default value for this type"""
        return cls._default_value
    
    def __call__(cls, *args, **kwargs):
        """Handle instantiation"""
        if not args and not kwargs and hasattr(cls, '_default_value'):
            # Return default value if no arguments provided
            return cls._default_value
        
        # Special handling for Bool class since bool objects can't have attributes
        if cls.__name__ == 'Bool':
            # Create a simple object that can hold the constraints
            instance = object.__new__(cls)
        else:
            # Create instance with constraint parameters
            # Only pass positional args to parent constructor, filter out our custom kwargs
            parent_args = args
            instance = super().__call__(*parent_args)
        
        # Set constraint parameters from kwargs or class defaults
        instance.answer_min = kwargs.get('answer_min', getattr(cls, '_answer_min', 0))
        instance.answer_max = kwargs.get('answer_max', getattr(cls, '_answer_max', 1000))
        instance.options = kwargs.get('options', getattr(cls, '_options', None))
        
        # Set annotation and constraints
        instance.annotation = getattr(cls, '_annotation', int)
        instance.constraints = Constraints(
            answer_min=instance.answer_min, 
            answer_max=instance.answer_max, 
            options=instance.options,
            custom_type_class=cls  # Store the custom type class
        )
        
        return instance


class Int(int, metaclass=CustomType, default=0, type_name='int', answer_min=0, answer_max=1000, options=None):
    _annotation = int
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(int))
    
    @classmethod
    def get_random(cls, answer_min=None, answer_max=None, options=None):
        return random.randint(answer_min, answer_max)

class Str(str, metaclass=CustomType, default='', type_name='str', answer_min=0, answer_max=1000, options=None):
    _annotation = str
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))

    @classmethod
    def get_random(cls, answer_min=None, answer_max=None, options=None):
        if options is not None:
            return random.choice(options)
        else:
            # generate random string of length between answer_min and answer_max
            return ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(answer_min, answer_max)))

class List(list, metaclass=CustomType, default=[], type_name='list', answer_min=0, answer_max=1000, options=None):
    _annotation = list
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(list))

    @classmethod
    def get_random(cls, answer_min=None, answer_max=None, options=None):
        if options is not None:
            return [random.choice(options)]
        else:
            return cls.default

class Float(float, metaclass=CustomType, default=0.0, type_name='float', answer_min=0, answer_max=1000, options=None):
    _annotation = float
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(float))
    
    @classmethod
    def get_random(cls, answer_min=None, answer_max=None, options=None):
        return random.uniform(answer_min, answer_max)

class Bool(metaclass=CustomType, default=False, type_name='bool', answer_min=0, answer_max=1, options=None):
    _annotation = bool
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(lambda x: bool(x), handler(bool))
    
    @classmethod
    def get_random(cls, answer_min=None, answer_max=None, options=None):
        return random.choice([True, False])

class Year(int, metaclass=CustomType, default=2025, type_name='year', answer_min=1900, answer_max=2025, options=None):
    _annotation = int
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(lambda x: int(x), handler(int))
    
    @classmethod
    def get_random(cls, answer_min=None, answer_max=None, options=None):
        return random.randint(answer_min, answer_max)

class URL(str, metaclass=CustomType, default='', type_name='url', answer_min=0, answer_max=1000, options=None):
    _annotation = str
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))
    
    @classmethod
    def get_random(cls, answer_min=None, answer_max=None, options=None):
        return f"https://www.example.com/{random.randint(1, 1000000)}"



def get_type(t: str):
    # Handle both capitalized custom type names and lowercase built-in type names
    type_mapping = {
        'Int': Int,
        'int': Int,
        'Str': Str, 
        'str': Str,
        'list': List,
        'List': List,
        'Float': Float,
        'float': Float,
        'Bool': Bool,
        'bool': Bool,
        'Year': Year,
        'URL': URL,
        'url': URL
    }
    
    if t in type_mapping:
        return type_mapping[t]
    else:
        raise ValueError(f"Invalid type is: {t}")
    

PRIMITIVE_TYPES = [Int, Str, List, Float, Bool, Year, URL]