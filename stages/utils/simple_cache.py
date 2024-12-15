import os
import json
import hashlib
from functools import wraps
from pathlib import Path
from typing import Any, Callable

def simple_cache(cache_dir: str | Path) -> Callable:
    """
    A decorator that caches function results in JSON files within the specified directory.
    
    Args:
        cache_dir: Directory path where cache files will be stored
        
    Returns:
        Decorated function that implements caching
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_file = cache_path / f"{cache_hash}.json"
            
            # Return cached result if it exists
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
                    
            # Calculate result and cache it
            result = func(*args, **kwargs)
            with open(cache_file, 'w') as f:
                json.dump(result, f)
                
            return result
            
        return wrapper
    return decorator
