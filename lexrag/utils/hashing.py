import hashlib

def compute_sha256(content: str | bytes) -> str:
    """Compute deterministic SHA-256 hash for content.
    
    Args:
        content: The text or bytes to hash.
        
    Returns:
        Hex digest of the SHA-256 hash.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()
