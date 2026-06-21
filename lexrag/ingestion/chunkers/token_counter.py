import tiktoken
from functools import lru_cache
from typing import List

@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    """Cache the encoding object since instantiating it is expensive."""
    return tiktoken.get_encoding("cl100k_base")

class TokenCounter:
    """Utility class to count tokens using tiktoken."""
    
    @staticmethod
    def count(text: str) -> int:
        """Count the number of tokens in the given text."""
        return len(get_encoding().encode(text))
        
    @staticmethod
    def split_to_fit(text: str, max_tokens: int) -> List[str]:
        """Split a long text into chunks that fit within max_tokens."""
        tokens = get_encoding().encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunks.append(get_encoding().decode(tokens[i:i + max_tokens]))
        return chunks
