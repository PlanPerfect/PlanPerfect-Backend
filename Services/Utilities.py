import uuid
import hashlib
import base64

"""
    Utilities is a simple service which provides general-purpose utility functions such as random ID generation, hashing, and encoding/decoding.
"""

def GenerateRandomID(length=None):
    random_id = str(uuid.uuid4()).replace('-', '')
    if length is not None:
        return random_id[:length]
    return random_id

def GenerateRandomInt(min_value=0, max_value=100):
    return uuid.uuid4().int % (max_value - min_value + 1) + min_value

def HashString(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()

def EncodeToBase64(input_string):
    encoded_bytes = base64.b64encode(input_string.encode('utf-8'))
    return encoded_bytes.decode('utf-8')

def DecodeFromBase64(encoded_string):
    decoded_bytes = base64.b64decode(encoded_string.encode('utf-8'))
    return decoded_bytes.decode('utf-8')