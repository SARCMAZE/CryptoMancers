import pandas as pd
from Crypto.Hash import SHA256, SHA512, MD5, SHA1, SHA224, SHA3_256, SHA3_512

def format_ciphertext(ciphertext):
    return ' '.join(ciphertext[i:i + 2].upper() for i in range(0, len(ciphertext), 2))

def hash_md5(plaintext):
    hash_obj = MD5.new(plaintext)
    return format_ciphertext(hash_obj.hexdigest()), 'MD5'

def hash_sha1(plaintext):
    hash_obj = SHA1.new(plaintext)
    return format_ciphertext(hash_obj.hexdigest()), 'SHA-1'

def hash_sha224(plaintext):
    hash_obj = SHA224.new(plaintext)
    return format_ciphertext(hash_obj.hexdigest()), 'SHA-224'

def hash_sha256(plaintext):
    hash_obj = SHA256.new(plaintext)
    return format_ciphertext(hash_obj.hexdigest()), 'SHA-256'

def hash_sha512(plaintext):
    hash_obj = SHA512.new(plaintext)
    return format_ciphertext(hash_obj.hexdigest()), 'SHA-512'

def hash_sha3_256(plaintext):
    hash_obj = SHA3_256.new(plaintext)
    return format_ciphertext(hash_obj.hexdigest()), 'SHA3-256'

def hash_sha3_512(plaintext):
    hash_obj = SHA3_512.new(plaintext)
    return format_ciphertext(hash_obj.hexdigest()), 'SHA3-512'

def create_hash_dataset(num_samples=1000):
    data = []
    
    for _ in range(num_samples):
        plaintext = get_random_bytes(16)
        
        for hash_func in [hash_md5, hash_sha1, hash_sha224, hash_sha256, hash_sha512, hash_sha3_256, hash_sha3_512]:
            ciphertext, algorithm = hash_func(plaintext)
            data.append([algorithm, plaintext.hex(), ciphertext])
    
    df = pd.DataFrame(data, columns=['Algorithm', 'Plaintext', 'Hash'])
    df.to_csv('hashes.csv', index=False)

create_hash_dataset(num_samples=1000)
