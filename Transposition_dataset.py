import pandas as pd
from Crypto.Random import get_random_bytes
from tqdm import tqdm  # Import tqdm for progress tracking

def format_ciphertext(ciphertext):
    return ' '.join(ciphertext[i:i + 2].upper() for i in range(0, len(ciphertext), 2))

def encrypt_columnar_transposition(plaintext_hex, key):
    key = list(key)
    sorted_key = sorted((k, i) for i, k in enumerate(key))
    key_length = len(key)
    num_rows = len(plaintext_hex) // key_length + (len(plaintext_hex) % key_length > 0)
    padded_plaintext = plaintext_hex.ljust(num_rows * key_length)
    matrix = [padded_plaintext[i:i + key_length] for i in range(0, len(padded_plaintext), key_length)]
    ciphertext = ''.join(matrix[row][key.index(col)] for col in sorted(key) for row in range(num_rows))
    return format_ciphertext(ciphertext), 'Columnar Transposition', key

def encrypt_double_columnar(plaintext_hex, key1, key2):
    first_pass = encrypt_columnar_transposition(plaintext_hex, key1)[0].replace(' ', '')
    return encrypt_columnar_transposition(first_pass, key2)

def encrypt_scytale(plaintext_hex, key):
    num_cols = key
    num_rows = (len(plaintext_hex) + num_cols - 1) // num_cols
    padded_plaintext = plaintext_hex.ljust(num_rows * num_cols)
    ciphertext = ''.join(padded_plaintext[row * num_cols + col] for col in range(num_cols) for row in range(num_rows))
    return format_ciphertext(ciphertext), 'Scytale', key

def create_transposition_dataset(num_samples=1000):
    data = []
    
    # Initialize tqdm with a total count and an update interval
    with tqdm(total=num_samples, desc="Generating Datasets", unit="samples") as pbar:
        for i in range(num_samples):
            plaintext = get_random_bytes(16).hex()
            
            for encrypt_func, args in [(encrypt_columnar_transposition, ('KEY1',)), 
                                       (encrypt_double_columnar, ('KEY1', 'KEY2')), 
                                       (encrypt_scytale, (4,))]:
                ciphertext, algorithm, key = encrypt_func(plaintext, *args)
                data.append([algorithm, plaintext, ciphertext, key])
            
            # Update the progress bar every 10,000 samples
            if (i + 1) % 10000 == 0:
                pbar.update(10000)
        
        # Final update if num_samples isn't a multiple of 10,000
        if num_samples % 10000 != 0:
            pbar.update(num_samples % 10000)
    
    df = pd.DataFrame(data, columns=['Algorithm', 'Plaintext', 'Ciphertext', 'Key'])
    df.to_csv('transposition_with_key.csv', index=False)

create_transposition_dataset(num_samples=1000000)
import pandas as pd
from Crypto.Random import get_random_bytes
from tqdm import tqdm  # Import tqdm for progress tracking

def format_ciphertext(ciphertext):
    return ' '.join(ciphertext[i:i + 2].upper() for i in range(0, len(ciphertext), 2))

def encrypt_columnar_transposition(plaintext_hex, key):
    key = list(key)
    sorted_key = sorted((k, i) for i, k in enumerate(key))
    key_length = len(key)
    num_rows = len(plaintext_hex) // key_length + (len(plaintext_hex) % key_length > 0)
    padded_plaintext = plaintext_hex.ljust(num_rows * key_length)
    matrix = [padded_plaintext[i:i + key_length] for i in range(0, len(padded_plaintext), key_length)]
    ciphertext = ''.join(matrix[row][key.index(col)] for col in sorted(key) for row in range(num_rows))
    return format_ciphertext(ciphertext), 'Columnar Transposition', key

def encrypt_double_columnar(plaintext_hex, key1, key2):
    first_pass = encrypt_columnar_transposition(plaintext_hex, key1)[0].replace(' ', '')
    return encrypt_columnar_transposition(first_pass, key2)

def encrypt_scytale(plaintext_hex, key):
    num_cols = key
    num_rows = (len(plaintext_hex) + num_cols - 1) // num_cols
    padded_plaintext = plaintext_hex.ljust(num_rows * num_cols)
    ciphertext = ''.join(padded_plaintext[row * num_cols + col] for col in range(num_cols) for row in range(num_rows))
    return format_ciphertext(ciphertext), 'Scytale', key

def create_transposition_dataset(num_samples=1000):
    data = []
    
    # Initialize tqdm with a total count and an update interval
    with tqdm(total=num_samples, desc="Generating Datasets", unit="samples") as pbar:
        for i in range(num_samples):
            plaintext = get_random_bytes(16).hex()
            
            for encrypt_func, args in [(encrypt_columnar_transposition, ('KEY1',)), 
                                       (encrypt_double_columnar, ('KEY1', 'KEY2')), 
                                       (encrypt_scytale, (4,))]:
                ciphertext, algorithm, key = encrypt_func(plaintext, *args)
                data.append([algorithm, plaintext, ciphertext, key])
            
            # Update the progress bar every 10,000 samples
            if (i + 1) % 10000 == 0:
                pbar.update(10000)
        
        # Final update if num_samples isn't a multiple of 10,000
        if num_samples % 10000 != 0:
            pbar.update(num_samples % 10000)
    
    df = pd.DataFrame(data, columns=['Algorithm', 'Plaintext', 'Ciphertext', 'Key'])
    df.to_csv('transposition_with_key.csv', index=False)

create_transposition_dataset(num_samples=1000000)
