import os
import random
import pandas as pd
from cryptography.hazmat.primitives.asymmetric import rsa, dsa, ec, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from tqdm import tqdm
import time  # Import time module to track progress time

# Function to generate RSA keys and "encrypt" plaintext
def rsa_encrypt(public_key_pem, plaintext):
    public_key = serialization.load_pem_public_key(public_key_pem, backend=default_backend())
    plaintext = plaintext.encode()  # Ensure it's in bytes
    ciphertext = public_key.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    # Format ciphertext in the desired format
    return ' '.join([f'{byte:02X}' for byte in ciphertext])

# Function to generate DSA keys and "sign" plaintext (simulating encryption)
def dsa_sign(private_key_pem, plaintext):
    private_key = serialization.load_pem_private_key(private_key_pem, password=None, backend=default_backend())
    signature = private_key.sign(
        plaintext.encode(),
        hashes.SHA256()
    )
    # Format signature as ciphertext in the desired format
    return ' '.join([f'{byte:02X}' for byte in signature])

# Function to "encrypt" data using EC (simulating with a signature)
def ec_sign(private_key_pem, plaintext):
    private_key = serialization.load_pem_private_key(private_key_pem, password=None, backend=default_backend())
    signature = private_key.sign(
        plaintext.encode(),
        ec.ECDSA(hashes.SHA256())
    )
    # Format signature as ciphertext in the desired format
    return ' '.join([f'{byte:02X}' for byte in signature])

# Function to generate RSA key pair
def generate_rsa_keys():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    public_key = private_key.public_key()
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private_key_pem, public_key_pem

# Function to generate DSA key pair
def generate_dsa_keys():
    private_key = dsa.generate_private_key(key_size=2048, backend=default_backend())
    public_key = private_key.public_key()
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private_key_pem, public_key_pem

# Function to generate EC key pair
def generate_ec_keys():
    private_key = ec.generate_private_key(ec.SECP256R1(), backend=default_backend())
    public_key = private_key.public_key()
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private_key_pem, public_key_pem

# Generate a random plaintext
def generate_plaintext():
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=16))

# Function to generate a dataset
def generate_dataset(num_samples=1000):
    data = []

    # Equal distribution for each algorithm
    algorithms = ['RSA', 'DSA', 'EC']
    num_per_algorithm = num_samples // len(algorithms)

    start_time = time.time()  # Start time for the whole dataset generation

    for algorithm_choice in algorithms:
        algorithm_start_time = time.time()  # Start time for each algorithm
        print(f"Generating {algorithm_choice} samples...")

        for _ in tqdm(range(num_per_algorithm), desc=f"Generating {algorithm_choice} samples"):
            plaintext = generate_plaintext()

            if algorithm_choice == 'RSA':
                private_key_pem, public_key_pem = generate_rsa_keys()
                ciphertext = rsa_encrypt(public_key_pem, plaintext)
            elif algorithm_choice == 'DSA':
                private_key_pem, public_key_pem = generate_dsa_keys()
                ciphertext = dsa_sign(private_key_pem, plaintext)
            elif algorithm_choice == 'EC':
                private_key_pem, public_key_pem = generate_ec_keys()
                ciphertext = ec_sign(private_key_pem, plaintext)

            # Add the entry to the dataset
            data.append({
                'Algorithm': algorithm_choice,
                'Plaintext': plaintext,
                'Ciphertext': ciphertext,
                'Public Key': public_key_pem.decode(),
                'Private Key': private_key_pem.decode()
            })

        # Print time taken for each algorithm
        algorithm_elapsed_time = time.time() - algorithm_start_time
        print(f"{algorithm_choice} generation completed in {algorithm_elapsed_time:.2f} seconds.")

    # Shuffle the dataset to remove any order
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle and reset the index

    # Print total time taken
    total_elapsed_time = time.time() - start_time
    print(f"Total dataset generation completed in {total_elapsed_time:.2f} seconds.")

    return df

# Generate the dataset
df = generate_dataset(num_samples=100000)  # 10 samples per algorithm for a balanced dataset

# Save the dataset with keys (public & private)
df_with_keys = df[['Algorithm', 'Plaintext', 'Ciphertext', 'Public Key', 'Private Key']]
df_with_keys.to_csv('asymmetric_encryption_dataset_with_keys.csv', index=False)

# Confirm successful generation
print("Dataset generated successfully with keys!")
