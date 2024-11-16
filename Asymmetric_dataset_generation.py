import pandas as pd
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
import random

# Function to generate random plaintext of random length
def generate_random_plaintext(min_length=8, max_length=32):
    length = random.randint(min_length, max_length)
    return get_random_bytes(length)

# Function to format ciphertext to uppercase with space after every two characters
def format_ciphertext(ciphertext):
    return ' '.join(ciphertext[i:i + 2].upper() for i in range(0, len(ciphertext), 2))

# RSA Encryption
def encrypt_rsa(plaintext):
    key = RSA.generate(2048)
    public_key = key.publickey().export_key().decode()
    private_key = key.export_key().decode()
    cipher = PKCS1_OAEP.new(key.publickey())
    ciphertext = cipher.encrypt(plaintext)
    return format_ciphertext(ciphertext.hex()), 'RSA', public_key, private_key

# Create asymmetric dataset with public and private keys
def create_asymmetric_dataset(num_samples=1000):
    data_with_keys = []
    data_without_keys = []
    
    for _ in range(num_samples):
        plaintext = generate_random_plaintext()
        
        # Encrypt with RSA
        try:
            ciphertext, algorithm, public_key, private_key = encrypt_rsa(plaintext)
            # Append with keys
            data_with_keys.append([algorithm, plaintext.hex(), ciphertext, public_key, private_key])
            # Append without keys
            data_without_keys.append([algorithm, plaintext.hex(), ciphertext])
        except Exception as e:
            print(f"Error with {algorithm}: {e}")
    
    # Save datasets to CSV files
    df_with_keys = pd.DataFrame(data_with_keys, columns=['Algorithm', 'Plaintext', 'Ciphertext', 'Public Key', 'Private Key'])
    df_without_keys = pd.DataFrame(data_without_keys, columns=['Algorithm', 'Plaintext', 'Ciphertext'])
    
    df_with_keys.to_csv('asymmetric_with_keys.csv', index=False)
    df_without_keys.to_csv('asymmetric_without_keys.csv', index=False)

# Generate the dataset
create_asymmetric_dataset(num_samples=10000)
