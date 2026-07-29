def base32_encode(data: bytes, padding: bool = True) -> str:
    """Encode bytes using RFC4648 base32 hex alphabet.
    
    Args:
        data: Bytes to encode
        padding: Whether to add padding
        
    Returns:
        Base32 encoded string
    """
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUV"
    result = ""
    bits = 0
    value = 0
    
    for byte in data:
        value = (value << 8) | byte
        bits += 8
        
        while bits >= 5:
            result += chars[(value >> (bits - 5)) & 31]
            bits -= 5
    
    if bits > 0:
        result += chars[(value << (5 - bits)) & 31]
    
    if padding:
        while len(result) % 8 != 0:
            result += "="
    
    return result
