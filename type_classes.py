from typing import Any
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

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
        return super().__call__(*args, **kwargs)


class Int(int, metaclass=CustomType, default=0, type_name='int'):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(int))

class Str(str, metaclass=CustomType, default='', type_name='str'):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))

class List(list, metaclass=CustomType, default=[], type_name='list'):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(list))

class Float(float, metaclass=CustomType, default=0.0, type_name='float'):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(float))

class Bool(metaclass=CustomType, default=False, type_name='bool'):
    def __new__(cls, value=None):
        if value is None:
            return cls._default_value
        return bool(value)
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(lambda x: bool(x), handler(bool))

class Year(metaclass=CustomType, default=2025, type_name='year'):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(lambda x: int(x), handler(int))

class URL(str, metaclass=CustomType, default='', type_name='url'):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))

def get_type(t: str):
    if t == 'Int':
        return Int
    elif t == 'Str':
        return Str
    elif t == 'list':
        return List
    elif t == 'Float':
        return Float
    elif t == 'Bool':
        return Bool
    elif t == 'Year':
        return Year
    elif t == 'URL':
        return URL
    else:
        raise ValueError(f"Invalid type is: {t}")
    

PRIMITIVE_TYPES = [Int, Str, List, Float, Bool, Year, URL]