import numpy as np
import pandas as pd
import random
from Crypto.Cipher import AES, DES, Blowfish, ARC4, ChaCha20, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Hash import SHA256, SHA512, MD5, SHA1, SHA224, SHA384

# Function to generate random plaintext of random length
def generate_random_plaintext(min_length=8, max_length=32):
    length = random.randint(min_length, max_length)  # Random length between min and max
    return get_random_bytes(length)

# Function to format ciphertext to uppercase with space after every two characters
def format_ciphertext(ciphertext):
    return ' '.join(ciphertext[i:i + 2].upper() for i in range(0, len(ciphertext), 2))

# Function to encrypt using AES
def encrypt_aes(plaintext):
    key = get_random_bytes(16)  # AES key size is 16 bytes for AES-128
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return format_ciphertext((cipher.nonce + tag + ciphertext).hex()), 'AES'

# Function to encrypt using DES
def encrypt_des(plaintext):
    key = get_random_bytes(8)  # DES key size is 8 bytes
    cipher = DES.new(key, DES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return format_ciphertext((cipher.nonce + tag + ciphertext).hex()), 'DES'

# Function to encrypt using Blowfish
def encrypt_blowfish(plaintext):
    key = get_random_bytes(16)  # Blowfish key size can be from 4 to 56 bytes
    cipher = Blowfish.new(key, Blowfish.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return format_ciphertext((cipher.nonce + tag + ciphertext).hex()), 'Blowfish'

# Function to encrypt using ChaCha20
def encrypt_chacha20(plaintext):
    key = get_random_bytes(32)  # ChaCha20 key size is 32 bytes
    cipher = ChaCha20.new(key=key)
    ciphertext = cipher.encrypt(plaintext)
    return format_ciphertext((cipher.nonce + ciphertext).hex()), 'ChaCha20'

# Function to encrypt using RSA
def encrypt_rsa(plaintext):
    key = RSA.generate(2048)
    public_key = key.publickey()
    cipher = PKCS1_OAEP.new(public_key)
    ciphertext = cipher.encrypt(plaintext)
    return format_ciphertext(ciphertext.hex()), 'RSA'

# Function to encrypt using RC4
def encrypt_rc4(plaintext):
    key = get_random_bytes(16)  # RC4 key size can be between 1 and 256 bytes
    cipher = ARC4.new(key)
    ciphertext = cipher.encrypt(plaintext)
    return format_ciphertext(ciphertext.hex()), 'RC4'

# Transposition cipher implementations
def encrypt_columnar_transposition(plaintext_hex, key):
    key = list(key)
    sorted_key = sorted((k, i) for i, k in enumerate(key))  # Sort the key and keep track of original indices
    key_length = len(key)
    num_rows = len(plaintext_hex) // key_length + (len(plaintext_hex) % key_length > 0)
    padded_plaintext = plaintext_hex.ljust(num_rows * key_length)
    
    # Create the matrix for columnar transposition
    matrix = [padded_plaintext[i:i + key_length] for i in range(0, len(padded_plaintext), key_length)]
    
    ciphertext = ''.join(matrix[row][key.index(col)] for col in sorted(key) for row in range(num_rows))
    return format_ciphertext(ciphertext), 'Columnar Transposition'

def encrypt_double_transposition(plaintext_hex, key1, key2):
    # First columnar transposition
    first_pass = encrypt_columnar_transposition(plaintext_hex, key1)[0]
    # Second columnar transposition
    return encrypt_columnar_transposition(first_pass.replace(' ', ''), key2)

def encrypt_rail_fence(plaintext_hex, key):
    fence = [['\n' for i in range(len(plaintext_hex))] for j in range(key)]
    dir_down = None
    row, col = 0, 0

    for i in range(len(plaintext_hex)):
        if row == 0 or row == key - 1:
            dir_down = not dir_down
        fence[row][col] = plaintext_hex[i]
        col += 1
        row += 1 if dir_down else -1
    
    ciphertext = ''.join([''.join(r) for r in fence]).replace('\n', '')
    return format_ciphertext(ciphertext), 'Rail Fence'

def encrypt_scytale(plaintext_hex, key):
    # The key is the number of columns
    num_cols = key
    num_rows = (len(plaintext_hex) + num_cols - 1) // num_cols
    padded_plaintext = plaintext_hex.ljust(num_rows * num_cols)
    
    ciphertext = ''
    for col in range(num_cols):
        for row in range(num_rows):
            ciphertext += padded_plaintext[row * num_cols + col]
    
    return format_ciphertext(ciphertext), 'Scytale'

# Hash functions
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

def hash_sha384(plaintext):
    hash_obj = SHA384.new(plaintext)
    return format_ciphertext(hash_obj.hexdigest()), 'SHA-384'

def hash_sha512(plaintext):
    hash_obj = SHA512.new(plaintext)
    return format_ciphertext(hash_obj.hexdigest()), 'SHA-512'

# Create a dataset
def create_dataset(num_samples=1000, min_length=8, max_length=32):
    data = []
    
    for _ in range(num_samples):
        plaintext = generate_random_plaintext(min_length, max_length)
        plaintext_hex = plaintext.hex()  # Convert plaintext to hexadecimal

        # Add symmetric encryptions
        data.append(list(encrypt_aes(plaintext)))
        data.append(list(encrypt_des(plaintext)))
        data.append(list(encrypt_blowfish(plaintext)))
        data.append(list(encrypt_chacha20(plaintext)))
        data.append(list(encrypt_rc4(plaintext)))
        
        # Add RSA encryption
        data.append(list(encrypt_rsa(plaintext)))
        
        # Add transposition ciphers using plaintext in hex
        data.append(list(encrypt_columnar_transposition(plaintext_hex, 'KEY1')))
        data.append(list(encrypt_double_transposition(plaintext_hex, 'KEY1', 'KEY2')))
        data.append(list(encrypt_rail_fence(plaintext_hex, 3)))
        data.append(list(encrypt_scytale(plaintext_hex, 4)))  # Example key for scytale
        
        # Add hash functions
        data.append(list(hash_md5(plaintext)))
        data.append(list(hash_sha1(plaintext)))
        data.append(list(hash_sha224(plaintext)))
        data.append(list(hash_sha256(plaintext)))
        data.append(list(hash_sha384(plaintext)))
        data.append(list(hash_sha512(plaintext)))
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Ciphertext', 'Algorithm'])
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Generate dataset and save to CSV
dataset = create_dataset(num_samples=100000)
dataset.to_csv('ciphertext_dataset.csv', index=False)
