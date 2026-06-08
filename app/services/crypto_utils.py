from cryptography.fernet import Fernet
import os

_fernet = Fernet(os.environ["API_KEY_ENCRYPT_SECRET"].encode())


def encrypt_key(raw: str) -> str:
    return _fernet.encrypt(raw.encode()).decode()


def decrypt_key(enc: str) -> str:
    return _fernet.decrypt(enc.encode()).decode()
