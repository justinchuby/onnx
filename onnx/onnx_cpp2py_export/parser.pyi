from typing import Tuple

def parse_model(model: str) -> Tuple[bool, bytes, bytes]:
    """
    Returns (success-flag, error-message, serialized-proto).
    If success-flag is true, then serialized-proto contains the parsed ModelProto.
    Otherwise, error-message contains a string describing the parse error.
    """
    ...

def parse_graph(graph: str) -> Tuple[bool, bytes, bytes]:
    """
    Returns (success-flag, error-message, serialized-proto).
    If success-flag is true, then serialized-proto contains the parsed GraphProto.
    Otherwise, error-message contains a string describing the parse error.
    """
    ...

def parse_function(function: str) -> Tuple[bool, bytes, bytes]:
    """
    Returns (success-flag, error-message, serialized-proto).
    If success-flag is true, then serialized-proto contains the parsed FunctionProto.
    Otherwise, error-message contains a string describing the parse error.
    """
    ...
