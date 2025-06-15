# stocksite/firebase_client.py
import os
import firebase_admin
from firebase_admin import credentials, firestore

# Path comes from an env var or .env
cred = credentials.Certificate(os.environ['FIREBASE_KEY_PATH'])
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()
