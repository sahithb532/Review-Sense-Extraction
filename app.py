import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import base64
from io import BytesIO
from collections import Counter, defaultdict
import json
import numpy as np
import asyncio
import requests
import os
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# NLP Libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except:
    nlp = None

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="FeedbackAI Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_STORE_DIR = Path("models_store")
MODEL_STORE_DIR.mkdir(exist_ok=True)


def current_timestamp():
    """Return local timestamp string for consistent storage."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Custom CSS for Lavender-Neon Theme
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Poppins:wght@300;400;600&display=swap');
    
    :root {
        --lavender: #E6E6FA;
        --neon-purple: #B19CD9;
        --neon-pink: #FF6EC7;
        --neon-blue: #00F0FF;
        --dark-bg: #1a0933;
        --card-bg: #2d1b4e;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0423 0%, #1a0933 50%, #2d1b4e 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes neon-glow {
        0%, 100% { text-shadow: 0 0 10px #B19CD9, 0 0 20px #B19CD9, 0 0 30px #B19CD9; }
        50% { text-shadow: 0 0 20px #FF6EC7, 0 0 30px #FF6EC7, 0 0 40px #FF6EC7; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #E6E6FA;
        animation: neon-glow 3s infinite;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0933 0%, #2d1b4e 100%);
        border-right: 2px solid #B19CD9;
        box-shadow: 5px 0 15px rgba(177, 156, 217, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #B19CD9 0%, #FF6EC7 100%);
        color: white;
        border: 2px solid #00F0FF;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        font-family: 'Orbitron', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(177, 156, 217, 0.5);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 30px rgba(255, 110, 199, 0.8), 0 0 40px rgba(0, 240, 255, 0.6);
        animation: pulse 1s infinite;
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background: rgba(45, 27, 78, 0.6);
        border: 2px solid #B19CD9;
        border-radius: 15px;
        color: #E6E6FA;
        padding: 12px;
        font-family: 'Poppins', sans-serif;
    }
    
    .card {
        background: linear-gradient(135deg, rgba(45, 27, 78, 0.8) 0%, rgba(26, 9, 51, 0.8) 100%);
        border: 2px solid #B19CD9;
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(177, 156, 217, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(255, 110, 199, 0.5);
        border-color: #FF6EC7;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        color: #00F0FF;
        font-size: 2em;
        text-shadow: 0 0 10px #00F0FF;
    }
    
    .aspect-positive {
        color: #00F0FF;
        font-weight: 600;
        background: rgba(0, 240, 255, 0.2);
        padding: 2px 8px;
        border-radius: 5px;
    }
    
    .aspect-negative {
        color: #FF6EC7;
        font-weight: 600;
        background: rgba(255, 110, 199, 0.2);
        padding: 2px 8px;
        border-radius: 5px;
    }
    
    .aspect-neutral {
        color: #B19CD9;
        font-weight: 600;
        background: rgba(177, 156, 217, 0.2);
        padding: 2px 8px;
        border-radius: 5px;
    }
    
    .uncertain-box {
        background: rgba(255, 110, 199, 0.15);
        border: 2px dashed #FF6EC7;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #B19CD9 0%, #FF6EC7 50%, #00F0FF 100%);
        padding: 10px;
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        color: white;
        font-weight: 700;
        box-shadow: 0 -5px 20px rgba(177, 156, 217, 0.5);
        z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True)

# Database Functions
def init_db():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  is_admin INTEGER DEFAULT 0,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Workspaces table
    c.execute('''CREATE TABLE IF NOT EXISTS workspaces
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  description TEXT,
                  created_by INTEGER NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (created_by) REFERENCES users (id))''')
    
    # Workspace members table
    c.execute('''CREATE TABLE IF NOT EXISTS workspace_members
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  workspace_id INTEGER NOT NULL,
                  user_id INTEGER NOT NULL,
                  role TEXT DEFAULT 'member',
                  added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (workspace_id) REFERENCES workspaces (id),
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Feedback table
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  workspace_id INTEGER NOT NULL,
                  user_id INTEGER NOT NULL,
                  feedback_text TEXT NOT NULL,
                  cleaned_text TEXT,
                  sentiment TEXT,
                  sentiment_score REAL,
                  confidence_score REAL,
                  aspects TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (workspace_id) REFERENCES workspaces (id),
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Models table
    c.execute('''CREATE TABLE IF NOT EXISTS models
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  workspace_id INTEGER NOT NULL,
                  version INTEGER NOT NULL,
                  accuracy REAL,
                  f1_score REAL,
                  trained_by INTEGER NOT NULL,
                  model_path TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (workspace_id) REFERENCES workspaces (id),
                  FOREIGN KEY (trained_by) REFERENCES users (id))''')
    
    # Corrections table (for active learning)
    c.execute('''CREATE TABLE IF NOT EXISTS corrections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  feedback_id INTEGER NOT NULL,
                  user_id INTEGER NOT NULL,
                  original_sentiment TEXT,
                  corrected_sentiment TEXT,
                  corrected_aspects TEXT,
                  remarks TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (feedback_id) REFERENCES feedback (id),
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Activity logs table
    c.execute('''CREATE TABLE IF NOT EXISTS activity_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  workspace_id INTEGER,
                  action_type TEXT NOT NULL,
                  description TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id),
                  FOREIGN KEY (workspace_id) REFERENCES workspaces (id))''')
    
    # Schema upgrades for new features
    def safe_add_column(table, column, definition):
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
    
    safe_add_column('feedback', 'predicted_intent', 'TEXT')
    safe_add_column('feedback', 'corrected_intent', 'TEXT')
    safe_add_column('feedback', 'predicted_entities', 'TEXT')
    safe_add_column('feedback', 'corrected_entities', 'TEXT')
    safe_add_column('feedback', 'raw_confidence', 'REAL DEFAULT 0.0')
    
    safe_add_column('corrections', 'original_intent', 'TEXT')
    safe_add_column('corrections', 'corrected_intent', 'TEXT')
    safe_add_column('corrections', 'original_entities', 'TEXT')
    safe_add_column('corrections', 'corrected_entities', 'TEXT')
    
    safe_add_column('models', 'model_type', "TEXT DEFAULT 'sentiment'")
    safe_add_column('models', 'notes', 'TEXT')
    
    # Active Learning columns
    safe_add_column('feedback', 'is_uncertain', 'BOOLEAN DEFAULT 0')
    safe_add_column('feedback', 'needs_review', 'BOOLEAN DEFAULT 0')
    safe_add_column('feedback', 'is_corrected', 'BOOLEAN DEFAULT 0')
    safe_add_column('feedback', 'corrected_sentiment', 'TEXT')
    safe_add_column('feedback', 'corrected_aspects', 'TEXT')
    safe_add_column('feedback', 'predicted_sentiment', 'TEXT')
    safe_add_column('feedback', 'predicted_aspects', 'TEXT')
    
    conn.commit()
    
    # Make first user admin
    c.execute("SELECT COUNT(*) FROM users")
    user_count = c.fetchone()[0]
    if user_count == 0:
        pass  # Will be set when first user registers
    
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, email, password):
    try:
        conn = sqlite3.connect('users.db', check_same_thread=False)
        c = conn.cursor()
        password_hash = hash_password(password)
        
        # Check if this is the first user
        c.execute("SELECT COUNT(*) FROM users")
        user_count = c.fetchone()[0]
        is_admin = 1 if user_count == 0 else 0
        
        c.execute("INSERT INTO users (username, email, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?, ?)",
                  (username, email, password_hash, is_admin, current_timestamp()))
        user_id = c.lastrowid
        
        # Create default workspace for new user
        c.execute("INSERT INTO workspaces (name, description, created_by) VALUES (?, ?, ?)",
                  (f"{username}'s Workspace", "Default workspace", user_id))
        workspace_id = c.lastrowid
        
        # Log activity
        c.execute("INSERT INTO activity_logs (user_id, workspace_id, action_type, description) VALUES (?, ?, ?, ?)",
                  (user_id, workspace_id, "register", f"User {username} registered"))
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    password_hash = hash_password(password)
    c.execute("SELECT id, username FROM users WHERE username=? AND password_hash=?",
              (username, password_hash))
    result = c.fetchone()
    conn.close()
    return result

# ========================== AI-POWERED ASPECT-BASED SENTIMENT ANALYSIS ==========================

# Configuration for AI-powered analysis
USE_AI_ANALYSIS = True  # Toggle to enable/disable AI analysis
CONFIDENCE_THRESHOLD = 0.75  # Updated to 75% (can be adjusted 70-80%)

# Comprehensive Aspect Dictionary with Synonyms
ASPECT_CATEGORIES = {
    'battery': ['battery', 'power', 'charge', 'charging', 'backup', 'drain', 'draining', 'juice', 'mah', 'battery life'],
    'camera': ['camera', 'photo', 'picture', 'photos', 'pictures', 'lens', 'zoom', 'focus', 'megapixel', 'mp', 'selfie', 'portrait', 'image', 'video'],
    'screen': ['screen', 'display', 'panel', 'brightness', 'resolution', 'bezels', 'notch', 'oled', 'amoled', 'lcd', 'touch'],
    'performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'lagging', 'responsive', 'smooth', 'processor', 'ram', 'multitask', 'freezing', 'hang'],
    'design': ['design', 'look', 'looks', 'appearance', 'aesthetic', 'style', 'sleek', 'elegant', 'premium', 'finish', 'beautiful', 'ugly'],
    'build': ['build', 'construction', 'material', 'materials', 'plasticky', 'metal', 'glass', 'body', 'durability', 'durable', 'sturdy', 'solid', 'build quality'],
    'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'affordable', 'pricing', 'money', 'overpriced', 'budget'],
    'service': ['service', 'support', 'customer', 'help', 'assistance', 'warranty', 'repair', 'replacement', 'customer service'],
    'delivery': ['delivery', 'shipping', 'ship', 'arrived', 'package', 'packaging', 'box', 'courier'],
    'sound': ['sound', 'audio', 'speaker', 'speakers', 'volume', 'loud', 'bass', 'treble', 'music', 'headphone', 'earphone', 'sound quality'],
    'size': ['size', 'compact', 'bulky', 'heavy', 'light', 'weight', 'portable', 'thickness', 'dimensions'],
    'features': ['feature', 'features', 'functionality', 'function', 'functions', 'capabilities', 'options'],
    'ui': ['ui', 'interface', 'software', 'os', 'android', 'ios', 'system', 'menu', 'navigation', 'bloatware', 'app'],
    'connectivity': ['wifi', 'bluetooth', 'network', 'signal', 'connection', 'connectivity', '4g', '5g', 'lte', 'internet'],
    'storage': ['storage', 'memory', 'space', 'gb', 'capacity', 'expandable', 'internal']
}

SENTIMENT_LABEL_SCORES = {
    'Negative': -1.0,
    'Neutral': 0.0,
    'Positive': 1.0
}

# Advanced Sentiment Lexicon with Intensifiers
POSITIVE_LEXICON = {
    # Strong positive (1.0)
    'excellent': 1.0, 'amazing': 1.0, 'fantastic': 1.0, 'outstanding': 1.0, 'superb': 1.0,
    'brilliant': 1.0, 'perfect': 1.0, 'exceptional': 1.0, 'phenomenal': 1.0, 'wonderful': 1.0,
    'marvelous': 1.0, 'spectacular': 1.0, 'magnificent': 1.0, 'impressive': 1.0, 'incredible': 1.0,
    'awesome': 1.0, 'terrific': 1.0, 'fabulous': 1.0, 'sensational': 1.0,
    
    # Moderate positive (0.7-0.9)
    'great': 0.9, 'good': 0.8, 'nice': 0.7, 'love': 0.9, 'loved': 0.9, 'loving': 0.9,
    'beautiful': 0.9, 'solid': 0.8, 'sturdy': 0.8, 'durable': 0.8, 'quality': 0.7,
    'premium': 0.9, 'superior': 0.9, 'better': 0.8, 'best': 1.0, 'recommend': 0.9,
    'clear': 0.7, 'crystal': 0.9, 'sharp': 0.8, 'bright': 0.8, 'vivid': 0.8,
    'smooth': 0.8, 'fast': 0.8, 'quick': 0.8, 'responsive': 0.8, 'powerful': 0.9,
    
    # Mild positive (0.5-0.6)
    'fine': 0.6, 'decent': 0.6, 'okay': 0.5, 'ok': 0.5, 'adequate': 0.6, 'acceptable': 0.6,
    'satisfied': 0.7, 'happy': 0.8, 'pleased': 0.7, 'enjoy': 0.8, 'liked': 0.7
}

NEGATIVE_LEXICON = {
    # Strong negative (1.0)
    'terrible': 1.0, 'horrible': 1.0, 'awful': 1.0, 'pathetic': 1.0, 'useless': 1.0,
    'worst': 1.0, 'hate': 1.0, 'hated': 1.0, 'disgusting': 1.0, 'abysmal': 1.0,
    'atrocious': 1.0, 'dreadful': 1.0, 'disastrous': 1.0, 'catastrophic': 1.0,
    'defective': 1.0, 'broken': 1.0, 'broke': 1.0, 'failed': 1.0, 'failure': 1.0,
    
    # Moderate negative (0.7-0.9)
    'bad': 0.9, 'poor': 0.9, 'disappointing': 0.9, 'disappointed': 0.9, 'worse': 0.8,
    'cheap': 0.8, 'flimsy': 0.9, 'weak': 0.8, 'fragile': 0.8, 'unstable': 0.8,
    'slow': 0.7, 'sluggish': 0.8, 'laggy': 0.8, 'lag': 0.7, 'lagging': 0.7,
    'dim': 0.7, 'dull': 0.7, 'blurry': 0.8, 'fuzzy': 0.7, 'grainy': 0.7,
    'cracked': 1.0, 'crack': 0.9, 'scratch': 0.7, 'scratched': 0.7, 'damaged': 0.9,
    'waste': 0.9, 'avoid': 0.9, 'regret': 0.9, 'disappointed': 0.9,
    
    # Mild negative (0.5-0.6)
    'mediocre': 0.7, 'average': 0.5, 'inadequate': 0.8, 'insufficient': 0.7,
    'issue': 0.7, 'issues': 0.8, 'problem': 0.8, 'problems': 0.9, 'concern': 0.6
}

# General product negative keywords (apply to all aspects if not specifically contradicted)
GENERAL_NEGATIVE_KEYWORDS = {
    'bad', 'terrible', 'horrible', 'awful', 'worst', 'pathetic', 'useless', 
    'disappointing', 'waste', 'avoid', 'regret', 'poor', 'defective'
}

# Intensifiers and Diminishers
INTENSIFIERS = {
    'very': 1.5, 'extremely': 1.8, 'highly': 1.5, 'incredibly': 1.7, 'absolutely': 1.8,
    'completely': 1.7, 'totally': 1.6, 'really': 1.4, 'so': 1.5, 'too': 1.6,
    'exceptionally': 1.8, 'remarkably': 1.6, 'particularly': 1.4, 'especially': 1.4
}

DIMINISHERS = {
    'slightly': 0.6, 'somewhat': 0.7, 'fairly': 0.7, 'rather': 0.7, 'quite': 0.8,
    'pretty': 0.7, 'kind': 0.6, 'sort': 0.6, 'little': 0.6, 'bit': 0.6
}

# Negation Words
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'none',
    'hardly', 'barely', 'scarcely', "n't", 'cannot', 'cant', 'without', 'lack',
    'lacking', 'lacks', 'doesnt', "don't", "didn't", "won't", "wouldn't", "shouldn't"
}

# Contrastive Conjunctions
CONTRASTIVE_CONJUNCTIONS = r'\b(but|however|although|though|yet|while|whereas|except|despite|nevertheless|nonetheless|still)\b'

INTENT_CATEGORIES = {
    "Battery Issue": ['battery', 'charge', 'charging', 'power', 'backup', 'mah'],
    "Camera Issue": ['camera', 'photo', 'selfie', 'picture', 'image'],
    "Screen Issue": ['screen', 'display', 'brightness', 'resolution', 'panel'],
    "Performance Issue": ['slow', 'lag', 'performance', 'speed', 'hang', 'freeze'],
    "Design Feedback": ['design', 'look', 'appearance', 'style'],
    "Price Concern": ['price', 'expensive', 'cheap', 'cost'],
    "Delivery Problem": ['delivery', 'shipping', 'courier', 'package'],
    "Service Feedback": ['service', 'support', 'customer', 'warranty'],
    "Audio Issue": ['sound', 'audio', 'speaker', 'volume'],
    "General Feedback": []
}

def advanced_aspect_extraction(text):
    """
    Advanced aspect extraction using multiple techniques:
    Only extract SPECIFIC product aspects that are explicitly mentioned
    
    CRITICAL: Ignore generic words like "product", "quality", "thing", "item"
    """
    text_lower = text.lower()
    found_aspects = {}
    
    # Generic non-aspect words to ignore
    GENERIC_WORDS = {
        'product', 'quality', 'thing', 'item', 'phone', 'device', 
        'good', 'bad', 'great', 'poor', 'excellent', 'terrible',
        'nice', 'awesome', 'horrible', 'amazing', 'worst', 'best'
    }
    
    # Method 1: Dictionary-based with synonyms - strict matching
    for aspect_name, keywords in ASPECT_CATEGORIES.items():
        for keyword in keywords:
            # Skip if keyword is a generic word
            if keyword in GENERIC_WORDS:
                continue
                
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                # Additional validation: make sure it's a real aspect mention
                # Check context to ensure it's not a false positive
                
                # For compound phrases like "battery life", "build quality", verify full phrase
                if ' ' in keyword:
                    # This is a multi-word aspect, it's more specific - accept it
                    if aspect_name not in found_aspects:
                        found_aspects[aspect_name] = []
                    if keyword not in found_aspects[aspect_name]:
                        found_aspects[aspect_name].append(keyword)
                else:
                    # Single word - verify it's a specific aspect word
                    # These are concrete product features/components
                    if aspect_name not in found_aspects:
                        found_aspects[aspect_name] = []
                    if keyword not in found_aspects[aspect_name]:
                        found_aspects[aspect_name].append(keyword)
    
    # Method 2: spaCy dependency parsing - only for concrete aspects
    if nlp is not None:
        try:
            doc = nlp(text)
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN']:
                    lemma = token.lemma_.lower()
                    # Skip generic words
                    if lemma in GENERIC_WORDS:
                        continue
                    
                    # Check if this is a specific aspect word
                    for aspect_name, keywords in ASPECT_CATEGORIES.items():
                        if lemma in keywords and lemma not in GENERIC_WORDS:
                            if aspect_name not in found_aspects:
                                found_aspects[aspect_name] = []
                            if lemma not in found_aspects[aspect_name]:
                                found_aspects[aspect_name].append(lemma)
        except:
            pass
    
    return found_aspects

def calculate_sentiment_with_modifiers(words, word_positions):
    """
    Calculate sentiment considering intensifiers, diminishers, and negations
    """
    sentiment_score = 0.0
    sentiment_weights = []
    
    for i, word in enumerate(words):
        base_score = 0.0
        modifier = 1.0
        
        # Check if word is in sentiment lexicon
        if word in POSITIVE_LEXICON:
            base_score = POSITIVE_LEXICON[word]
        elif word in NEGATIVE_LEXICON:
            base_score = -NEGATIVE_LEXICON[word]
        
        if base_score != 0:
            # Look for intensifiers/diminishers in previous 2 words
            for j in range(max(0, i-2), i):
                if words[j] in INTENSIFIERS:
                    modifier *= INTENSIFIERS[words[j]]
                elif words[j] in DIMINISHERS:
                    modifier *= DIMINISHERS[words[j]]
            
            # Look for negations in previous 3 words
            negated = False
            for j in range(max(0, i-3), i):
                if words[j] in NEGATION_WORDS:
                    negated = True
                    break
            
            if negated:
                base_score = -base_score * 0.8
            
            final_score = base_score * modifier
            sentiment_weights.append(final_score)
    
    return sentiment_weights

def advanced_aspect_sentiment(text, aspect_keywords):
    """
    Advanced aspect-specific sentiment analysis with:
    1. Clause-level analysis
    2. Intensifier/Diminisher detection
    3. Negation handling
    4. Contrastive conjunction awareness
    5. Distance-weighted opinion words
    6. General negative context detection
    """
    sentences = sent_tokenize(text)
    aspect_sentiments = []
    
    # Check for general negative sentiment affecting all aspects
    text_lower = text.lower()
    general_negative_phrases = [
        'bad product', 'terrible product', 'worst product', 'horrible product',
        'bad phone', 'terrible phone', 'worst phone', 'horrible phone',
        'not recommend', 'waste of money', 'don\'t buy', 'avoid this',
        'very bad', 'really bad', 'totally bad', 'completely bad'
    ]
    
    has_general_negative = any(phrase in text_lower for phrase in general_negative_phrases)
    
    # Check if aspect is explicitly mentioned positively despite general negativity
    aspect_explicitly_positive = False
    for keyword in aspect_keywords:
        positive_patterns = [
            f'{keyword} is good', f'{keyword} is great', f'{keyword} is excellent',
            f'{keyword} is amazing', f'good {keyword}', f'great {keyword}',
            f'excellent {keyword}', f'love the {keyword}', f'{keyword} is nice'
        ]
        if any(pattern in text_lower for pattern in positive_patterns):
            aspect_explicitly_positive = True
            break
    
    for sentence in sentences:
        # Check if sentence contains aspect
        sentence_lower = sentence.lower()
        aspect_found = False
        for keyword in aspect_keywords:
            if keyword in sentence_lower:
                aspect_found = True
                break
        
        if not aspect_found:
            continue
        
        # Split by contrastive conjunctions
        clauses = re.split(CONTRASTIVE_CONJUNCTIONS, sentence, flags=re.IGNORECASE)
        
        for clause in clauses:
            if not any(keyword in clause.lower() for keyword in aspect_keywords):
                continue
            
            # Tokenize clause
            words = word_tokenize(clause.lower())
            words = [w for w in words if w.isalpha()]
            
            # Find aspect position
            aspect_positions = []
            for i, word in enumerate(words):
                if word in aspect_keywords:
                    aspect_positions.append(i)
            
            if not aspect_positions:
                continue
            
            # Calculate sentiment with modifiers
            sentiment_weights = calculate_sentiment_with_modifiers(words, aspect_positions)
            
            if sentiment_weights:
                # Weight by distance from aspect
                aspect_center = np.mean(aspect_positions)
                weighted_sentiments = []
                
                for i, weight in enumerate(sentiment_weights):
                    # Find closest opinion word to aspect
                    distance = min(abs(i - pos) for pos in aspect_positions)
                    # Apply distance decay (closer opinions weighted more)
                    distance_weight = 1.0 / (1.0 + 0.1 * distance)
                    weighted_sentiments.append(weight * distance_weight)
                
                clause_sentiment = np.mean(weighted_sentiments)
                aspect_sentiments.append(clause_sentiment)
    
    # If no specific sentiment found but general negative context exists
    if not aspect_sentiments and has_general_negative and not aspect_explicitly_positive:
        return "Negative", -0.6, 0.70
    
    if not aspect_sentiments:
        return None, 0.0, 0.0
    
    # Calculate final scores
    avg_sentiment = np.mean(aspect_sentiments)
    std_sentiment = np.std(aspect_sentiments) if len(aspect_sentiments) > 1 else 0
    
    # Apply general negative bias if present and aspect not explicitly positive
    if has_general_negative and not aspect_explicitly_positive and avg_sentiment > -0.3:
        avg_sentiment = min(avg_sentiment - 0.4, -0.3)
    
    # Confidence based on consistency and magnitude
    consistency = 1.0 - min(std_sentiment / (abs(avg_sentiment) + 0.1), 1.0)
    magnitude = min(abs(avg_sentiment), 1.0)
    confidence = (consistency * 0.5 + magnitude * 0.5)
    confidence = max(0.5, min(confidence, 0.99))
    
    # Determine sentiment label with refined thresholds
    if avg_sentiment > 0.25:
        sentiment = "Positive"
        confidence = max(confidence, 0.65)
    elif avg_sentiment < -0.25:
        sentiment = "Negative"
        confidence = max(confidence, 0.65)
    else:
        sentiment = "Neutral"
        confidence = max(confidence, 0.5)
    
    return sentiment, avg_sentiment, confidence

async def ai_powered_absa(text):
    """
    AI-Powered Aspect-Based Sentiment Analysis using Claude API
    Provides highly accurate aspect detection and sentiment classification
    """
    try:
        # First, try rule-based extraction
        aspects_dict = advanced_aspect_extraction(text)
        
        # If no aspects found, use AI to intelligently extract aspects
        if not aspects_dict:
            # Let AI determine if there are implicit aspects mentioned
            analysis_prompt = f"""Analyze this customer feedback and identify product aspects mentioned explicitly OR implicitly.

Customer Feedback: "{text}"

Available product aspects: battery, camera, screen, performance, design, build, price, service, delivery, sound, size, features, ui, connectivity, storage

RULES:
1. Extract aspects that are EXPLICITLY mentioned (e.g., "battery is bad" → battery)
2. For GENERAL feedback about the product/quality, map to "build" aspect (e.g., "product is bad", "quality is poor" → build)
3. For OVERALL product complaints without specifics, infer the most likely aspect (e.g., "phone is terrible" → build)
4. If truly no aspects can be determined, return empty array

For each aspect found, determine:
1. Sentiment (Positive/Negative/Neutral)
2. Sentiment score (-1.0 to 1.0)
3. Confidence (0.5 to 0.99)
4. Brief reasoning

Return ONLY valid JSON array:
[
  {{
    "aspect": "aspect_name",
    "sentiment": "Positive|Negative|Neutral",
    "score": 0.0,
    "confidence": 0.0,
    "reasoning": "explanation"
  }}
]

If no aspects at all, return: []"""

            def make_analysis_call():
                return requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 2000,
                        "messages": [{"role": "user", "content": analysis_prompt}]
                    }
                )
            
            response = await asyncio.to_thread(make_analysis_call)
            
            if response.status_code == 200:
                data = response.json()
                content_text = ''.join([
                    block.get('text', '') 
                    for block in data.get('content', []) 
                    if block.get('type') == 'text'
                ])
                
                content_text = content_text.strip().replace('```json', '').replace('```', '').strip()
                ai_results = json.loads(content_text)
                
                # Add keywords based on AI extracted aspects
                for result in ai_results:
                    aspect_name = result['aspect']
                    if aspect_name in ASPECT_CATEGORIES:
                        result['keywords'] = [aspect_name]
                    else:
                        result['keywords'] = [aspect_name]
                
                return ai_results
            else:
                return []
        
        # If rule-based found aspects, analyze them with AI
        aspect_list = list(aspects_dict.keys())
        verified_aspects = []
        for aspect in aspect_list:
            keywords = aspects_dict[aspect]
            if any(keyword in text.lower() for keyword in keywords):
                verified_aspects.append(aspect)
        
        if not verified_aspects:
            return []
        
        prompt = f"""You are an expert in aspect-based sentiment analysis. Analyze the following customer feedback and determine the sentiment for the mentioned aspects.

Customer Feedback: "{text}"

Detected Aspects: {', '.join(verified_aspects)}

CRITICAL RULES:
1. ONLY analyze aspects that are EXPLICITLY mentioned in the feedback text
2. Determine the exact sentiment (Positive/Negative/Neutral)
3. Calculate sentiment score from -1.0 (very negative) to +1.0 (very positive)
4. Provide confidence level (0.5 to 0.99)

Consider:
- Context and nuance in the feedback
- Negations (e.g., "not good" is negative)
- Intensifiers (e.g., "very", "extremely")
- What opinion words directly modify each aspect
- General sentiment vs aspect-specific sentiment

Return ONLY a JSON array with aspects that are EXPLICITLY mentioned:
[
  {{
    "aspect": "aspect_name",
    "sentiment": "Positive|Negative|Neutral",
    "score": 0.0,
    "confidence": 0.0,
    "reasoning": "brief explanation"
  }}
]"""

        def make_api_call():
            return requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}]
            }
            )
        
        response = await asyncio.to_thread(make_api_call)
        
        if response.status_code == 200:
            data = response.json()
            content_text = ''.join([
                block.get('text', '') 
                for block in data.get('content', []) 
                if block.get('type') == 'text'
            ])
            
            content_text = content_text.strip().replace('```json', '').replace('```', '').strip()
            ai_results = json.loads(content_text)
            
            for result in ai_results:
                aspect_name = result['aspect']
                if aspect_name in aspects_dict:
                    result['keywords'] = aspects_dict[aspect_name]
                else:
                    result['keywords'] = [aspect_name]
            
            return ai_results
        else:
            return None
            
    except Exception as e:
        print(f"❌ AI Analysis Error: {str(e)}")
        return None

def perform_absa(text, use_ai=None):
    """
    Complete Advanced Aspect-Based Sentiment Analysis
    Uses AI-powered analysis when available, falls back to rule-based
    """
    # Use global setting if not specified
    if use_ai is None:
        use_ai = USE_AI_ANALYSIS
    
    # Extract aspects
    aspects_dict = advanced_aspect_extraction(text)
    
    if not aspects_dict:
        return []
    
    # Try AI-powered analysis first if enabled
    if use_ai:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ai_results = loop.run_until_complete(ai_powered_absa(text))
            loop.close()
            
            if ai_results:
                print("✅ Using AI-Powered ABSA")
                return ai_results
        except Exception as e:
            print(f"⚠️ AI analysis failed: {str(e)}, using rule-based fallback")
    
    # Fallback to enhanced rule-based analysis
    aspect_results = []
    
    for aspect_name, keywords in aspects_dict.items():
        sentiment, score, confidence = advanced_aspect_sentiment(text, keywords)
        
        if sentiment:
            aspect_results.append({
                'aspect': aspect_name,
                'keywords': keywords,
                'sentiment': sentiment,
                'score': score,
                'confidence': confidence
            })
    
    # Sort by confidence (most confident first)
    aspect_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return aspect_results

def save_feedback_with_aspects(
        user_id,
        feedback_text,
        cleaned_text,
        sentiment,
        sentiment_score,
        confidence,
        aspects,
        workspace_id=None,
        predicted_intent=None,
        predicted_entities=None,
        raw_confidence=None):
    """Save feedback with aspect analysis"""
    if workspace_id is None:
        # Get default workspace
        conn = sqlite3.connect('users.db', check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT id FROM workspaces WHERE created_by=? LIMIT 1", (user_id,))
        result = c.fetchone()
        conn.close()
        if result:
            workspace_id = result[0]
        else:
            return  # No workspace available
    
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    
    # Check if prediction is uncertain based on confidence threshold
    is_uncertain = confidence < CONFIDENCE_THRESHOLD
    needs_review = is_uncertain
    
    # Use datetime.now() to get correct local time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    c.execute("""INSERT INTO feedback 
                 (workspace_id, user_id, feedback_text, cleaned_text, sentiment, 
                  sentiment_score, confidence_score, aspects, created_at,
                  predicted_intent, predicted_entities, raw_confidence,
                  predicted_sentiment, predicted_aspects, is_uncertain, needs_review) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (workspace_id, user_id, feedback_text, cleaned_text, sentiment, 
               sentiment_score, confidence, json.dumps(aspects), current_time,
               predicted_intent, predicted_entities, raw_confidence,
               sentiment, json.dumps(aspects), is_uncertain, needs_review))
    conn.commit()
    conn.close()

def get_uncertain_feedback(user_id, workspace_id=None, confidence_threshold=None):
    """Get feedback that needs review (confidence < threshold)"""
    if confidence_threshold is None:
        confidence_threshold = CONFIDENCE_THRESHOLD
    
    conn = sqlite3.connect('users.db', check_same_thread=False)
    if workspace_id:
        df = pd.read_sql_query(
            """SELECT * FROM feedback 
               WHERE workspace_id=? AND needs_review=1 AND (is_corrected=0 OR is_corrected IS NULL)
               ORDER BY created_at DESC""",
            conn, params=(workspace_id,))
    else:
        df = pd.read_sql_query(
            """SELECT * FROM feedback 
               WHERE user_id=? AND needs_review=1 AND (is_corrected=0 OR is_corrected IS NULL)
               ORDER BY created_at DESC""",
            conn, params=(user_id,))
    conn.close()
    return df

def get_user_feedback(user_id, workspace_id=None):
    conn = sqlite3.connect('users.db', check_same_thread=False)
    if workspace_id:
        df = pd.read_sql_query(
            "SELECT * FROM feedback WHERE workspace_id=? ORDER BY created_at DESC",
            conn, params=(workspace_id,))
    else:
        df = pd.read_sql_query(
            "SELECT * FROM feedback WHERE user_id=? ORDER BY created_at DESC",
            conn, params=(user_id,))
    conn.close()
    return df

def delete_feedback(feedback_id, user_id):
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("DELETE FROM feedback WHERE id=? AND user_id=?", (feedback_id, user_id))
    conn.commit()
    conn.close()

def preprocess_text(text):
    """Basic text preprocessing"""
    try:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)
    except:
        return text.lower()

def lexicon_sentiment(text):
    """Enhanced sentiment analysis with better accuracy for short texts"""
    text_lower = text.lower()
    words = word_tokenize(text_lower)
    words = [w for w in words if w.isalpha()]
    
    # Lexicon-based scoring with negation handling
    sentiment_scores = []
    
    for i, word in enumerate(words):
        base_score = 0.0
        
        if word in POSITIVE_LEXICON:
            base_score = POSITIVE_LEXICON[word]
        elif word in NEGATIVE_LEXICON:
            base_score = -NEGATIVE_LEXICON[word]
        
        if base_score != 0:
            # Check negation
            negated = any(words[j] in NEGATION_WORDS for j in range(max(0, i-3), i))
            if negated:
                base_score = -base_score * 0.8
            
            # Check intensifiers/diminishers
            for j in range(max(0, i-2), i):
                if words[j] in INTENSIFIERS:
                    base_score *= INTENSIFIERS[words[j]]
                elif words[j] in DIMINISHERS:
                    base_score *= DIMINISHERS[words[j]]
            
            sentiment_scores.append(base_score)
    
    # Calculate final polarity
    if sentiment_scores:
        polarity = np.mean(sentiment_scores)
        confidence = min(abs(polarity) * 1.2, 0.95)
    else:
        # Fallback to TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        confidence = abs(polarity)
    
    # Classify sentiment
    if polarity > 0.15:
        return "Positive", polarity, max(confidence, 0.60)
    elif polarity < -0.15:
        return "Negative", polarity, max(confidence, 0.60)
    else:
        return "Neutral", polarity, max(confidence, 0.50)


MODEL_CACHE = {}


def predict_intent_and_entities(text):
    """Heuristic intent + entity extraction for feedback module."""
    text_lower = text.lower()
    detected_intent = "General Feedback"
    for intent, keywords in INTENT_CATEGORIES.items():
        if not keywords:
            continue
        if any(keyword in text_lower for keyword in keywords):
            detected_intent = intent
            break
    
    entities = []
    if nlp is not None:
        try:
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents]
        except Exception:
            entities = []
    
    if not entities:
        try:
            tokens = word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            # Keep nouns/proper nouns as pseudo-entities
            entities = [word for word, pos in tagged if pos in ('NN', 'NNS', 'NNP', 'NNPS')]
        except Exception:
            entities = []
    
    entities = list(dict.fromkeys(entities))  # preserve order, remove duplicates
    entity_str = ', '.join(entities[:5]) if entities else None
    return detected_intent, entity_str


def get_model_file(workspace_id):
    return MODEL_STORE_DIR / f"workspace_{workspace_id}_sentiment.pkl"


def load_sentiment_model(workspace_id):
    if not workspace_id:
        return None
    model_path = get_model_file(workspace_id)
    if not model_path.exists():
        return None
    cached = MODEL_CACHE.get(workspace_id)
    mtime = model_path.stat().st_mtime
    if cached and cached["mtime"] == mtime:
        return cached["model"]
    model = joblib.load(model_path)
    MODEL_CACHE[workspace_id] = {"model": model, "mtime": mtime}
    return model


def get_sentiment_prediction(text, workspace_id=None):
    """
    Returns sentiment label, score, display confidence, and raw confidence margin.
    Prefers trained model; falls back to lexicon.
    """
    model = load_sentiment_model(workspace_id)
    if model is not None:
        probs = model.predict_proba([text])[0]
        classes = model.classes_
        top_idx = int(np.argmax(probs))
        label = classes[top_idx]
        top_prob = float(probs[top_idx])
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else top_prob
        sentiment_score = 0.0
        for cls_label, prob in zip(classes, probs):
            sentiment_score += SENTIMENT_LABEL_SCORES.get(cls_label, 0.0) * float(prob)
        ui_confidence = max(top_prob, 0.55)
        return label, sentiment_score, ui_confidence, margin
    
    # Fallback to lexicon approach
    label, score, confidence = lexicon_sentiment(text)
    margin = min(abs(score), 1.0)
    return label, score, confidence, margin


def train_sentiment_model(workspace_id, trained_by, min_samples=30):
    """Train TF-IDF + LogisticRegression model for a workspace."""
    conn = sqlite3.connect('users.db', check_same_thread=False)
    df = pd.read_sql_query(
        "SELECT feedback_text, sentiment FROM feedback WHERE workspace_id=? AND sentiment IS NOT NULL",
        conn, params=(workspace_id,))
    conn.close()
    
    df = df.dropna(subset=['feedback_text', 'sentiment'])
    if len(df) < min_samples or df['sentiment'].nunique() < 2:
        return {
            "trained": False,
            "reason": f"Need at least {min_samples} samples spanning ≥2 sentiment classes"
        }
    
    X = df['feedback_text'].astype(str).tolist()
    y = df['sentiment'].tolist()
    
    test_size = 0.2 if len(df) >= 50 else 0.15
    stratify = y if len(set(y)) > 1 else None
    
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
    except ValueError:
        # Not enough samples to stratify
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=300, multi_class='auto'))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    accuracy = float(accuracy_score(y_val, y_pred))
    f1 = float(f1_score(y_val, y_pred, average='macro'))
    
    model_path = get_model_file(workspace_id)
    joblib.dump(pipeline, model_path)
    MODEL_CACHE.pop(workspace_id, None)
    
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    c.execute(
        "SELECT COALESCE(MAX(version), 0) FROM models WHERE workspace_id=? AND model_type='sentiment'",
        (workspace_id,))
    version = c.fetchone()[0] + 1
    c.execute("""INSERT INTO models (workspace_id, version, accuracy, f1_score, trained_by, 
                 model_path, model_type, notes)
                 VALUES (?, ?, ?, ?, ?, ?, 'sentiment', ?)""",
              (workspace_id, version, accuracy, f1, trained_by, str(model_path),
               f"Auto-trained on {len(df)} samples"))
    conn.commit()
    conn.close()
    
    log_activity(trained_by, workspace_id, "retrain", f"Retrained sentiment model v{version}")
    
    return {
        "trained": True,
        "accuracy": accuracy,
        "f1": f1,
        "version": version,
        "samples": len(df),
        "model_path": str(model_path)
    }


def detect_feedback_column(df):
    ordered = ["feedback_text", "feedback", "text", "review"]
    for col in ordered:
        if col in df.columns:
            return col
    return df.columns[0]


def bulk_import_feedback(csv_path, workspace_id, user_id, reset=False, use_ai=None):
    """Load a CSV file into a workspace and run analysis for each row."""
    if not os.path.exists(csv_path):
        return {"success": False, "message": f"File not found: {csv_path}"}
    
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"success": False, "message": f"No data found in {csv_path}"}
    
    text_col = detect_feedback_column(df)
    df[text_col] = df[text_col].astype(str).fillna("")
    
    conn = sqlite3.connect('users.db', check_same_thread=False)
    if reset:
        conn.execute("DELETE FROM feedback WHERE workspace_id=?", (workspace_id,))
    conn.commit()
    conn.close()
    
    imported = 0
    for text in df[text_col]:
        text = text.strip()
        if not text:
            continue
        cleaned_text = preprocess_text(text)
        sentiment, score, confidence, raw_conf = get_sentiment_prediction(text, workspace_id)
        aspects = perform_absa(text, use_ai=use_ai)
        predicted_intent, predicted_entities = predict_intent_and_entities(text)
        save_feedback_with_aspects(
            user_id,
            text,
            cleaned_text,
            sentiment,
            score,
            confidence,
            aspects,
            workspace_id=workspace_id,
            predicted_intent=predicted_intent,
            predicted_entities=predicted_entities,
            raw_confidence=raw_conf
        )
        imported += 1
    
    log_activity(user_id, workspace_id, "upload", f"Bulk imported {imported} rows from {os.path.basename(csv_path)}")
    return {"success": True, "imported": imported}


def bulk_train_from_csv(csv_path, workspace_name, user_id, reset_workspace=True, min_samples=30):
    """Helper to import a CSV into a dedicated workspace and train a model."""
    workspace_id = get_or_create_workspace_by_name(user_id, workspace_name, description=f"Auto workspace for {workspace_name}")
    import_result = bulk_import_feedback(csv_path, workspace_id, user_id, reset=reset_workspace)
    
    if not import_result.get("success"):
        return {"trained": False, "message": import_result.get("message", "Import failed.")}
    
    train_result = train_sentiment_model(workspace_id, user_id, min_samples=min_samples)
    train_result["workspace_id"] = workspace_id
    train_result["imported"] = import_result.get("imported", 0)
    return train_result
def create_wordcloud(text):
    try:
        wordcloud = WordCloud(
            width=800, height=400, 
            background_color='#1a0933',
            colormap='twilight',
            contour_color='#B19CD9',
            contour_width=2,
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        fig.patch.set_facecolor('#1a0933')
        return fig
    except:
        return None

# Authentication Pages
def login_page():
    st.markdown("<h1 style='text-align: center;'>🎯 FeedbackAI Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #B19CD9; font-size: 1.2em;'>AI-Powered Aspect-Based Sentiment Analysis</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Sign In", "✨ Sign Up"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Sign In", use_container_width=True):
                if username and password:
                    user = verify_user(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user[0]
                        st.session_state.username = user[1]
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials!")
                else:
                    st.warning("⚠️ Please fill all fields!")
        
        with tab2:
            st.markdown("### Create Your Account")
            new_username = st.text_input("Username", key="signup_username")
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            if st.button("Sign Up", use_container_width=True):
                if new_username and new_email and new_password and confirm_password:
                    if new_password == confirm_password:
                        if len(new_password) >= 6:
                            if create_user(new_username, new_email, new_password):
                                st.success("✅ Account created! Please sign in.")
                            else:
                                st.error("❌ Username or email already exists!")
                        else:
                            st.warning("⚠️ Password must be at least 6 characters!")
                    else:
                        st.error("❌ Passwords don't match!")
                else:
                    st.warning("⚠️ Please fill all fields!")
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_home():
    """Home Dashboard Overview"""
    st.markdown("<h1 style='text-align: center;'>🏠 Dashboard Overview</h1>", unsafe_allow_html=True)
    
    workspace_id = st.session_state.get('current_workspace')
    if not workspace_id:
        st.warning("⚠️ Please select a workspace first!")
        return
    
    df = get_user_feedback(st.session_state.user_id, workspace_id)
    uncertain_df = get_uncertain_feedback(st.session_state.user_id, workspace_id)
    
    # Overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Total Feedback", len(df))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        positive = len(df[df['sentiment'] == 'Positive']) if len(df) > 0 else 0
        st.metric("Positive", positive)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        neutral = len(df[df['sentiment'] == 'Neutral']) if len(df) > 0 else 0
        st.metric("Neutral", neutral)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        negative = len(df[df['sentiment'] == 'Negative']) if len(df) > 0 else 0
        st.metric("Negative", negative)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col5:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Needs Review", len(uncertain_df), delta=f"{CONFIDENCE_THRESHOLD*100:.0f}% threshold")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if len(df) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sentiment distribution chart
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📊 Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.4,
                marker=dict(colors=['#00F0FF', '#B19CD9', '#FF6EC7']),
                textfont=dict(size=16, color='white')
            )])
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E6E6FA', size=14),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_chart2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🎯 Top Aspects Mentioned")
            
            # Extract top aspects
            aspect_counts = defaultdict(int)
            for idx, row in df.iterrows():
                if 'aspects' in row and row['aspects']:
                    try:
                        aspects = json.loads(row['aspects'])
                        for asp in aspects:
                            aspect_counts[asp['aspect']] += 1
                    except:
                        pass
            
            if aspect_counts:
                top_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for aspect, count in top_aspects:
                    st.markdown(f"**{aspect.title()}**: {count} mentions")
            else:
                st.info("No aspect data available")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Recent feedback
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📝 Recent Feedback")
        recent_cols = ['feedback_text', 'sentiment', 'confidence_score', 'created_at']
        if 'is_uncertain' in df.columns:
            recent_cols.insert(3, 'is_uncertain')
        recent_df = df.head(5)[recent_cols].copy()
        if 'confidence_score' in recent_df.columns:
            recent_df['confidence_score'] = recent_df['confidence_score'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        if 'is_uncertain' in recent_df.columns:
            recent_df['is_uncertain'] = recent_df['is_uncertain'].apply(lambda x: "⚠️ Yes" if x else "✅ No")
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("📭 No feedback data available. Upload some feedback to get started!")# Main Dashboard
def dashboard():
    with st.sidebar:
        st.markdown(f"<h2 style='text-align: center;'>👤 {st.session_state.username}</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Workspace selector
        conn = sqlite3.connect('users.db', check_same_thread=False)
        workspaces = pd.read_sql_query(
            "SELECT id, name FROM workspaces WHERE created_by=? OR id IN (SELECT workspace_id FROM workspace_members WHERE user_id=?)",
            conn, params=(st.session_state.user_id, st.session_state.user_id))
        conn.close()
        
        if len(workspaces) > 0:
            workspace_names = workspaces['name'].tolist()
            if 'current_workspace' not in st.session_state:
                st.session_state.current_workspace = workspaces.iloc[0]['id']
            
            current_workspace_name = workspaces[workspaces['id'] == st.session_state.current_workspace]['name'].values
            if len(current_workspace_name) > 0:
                selected_workspace = st.selectbox(
                    "📁 Workspace:",
                    workspace_names,
                    index=workspace_names.index(current_workspace_name[0])
                )
                st.session_state.current_workspace = workspaces[workspaces['name'] == selected_workspace]['id'].values[0]
        else:
            st.warning("No workspace available")
            if st.button("➕ Create Workspace"):
                st.session_state.show_create_workspace = True
        
        st.markdown("---")
        
        page = st.radio("Navigation", 
                       ["🏠 Home", "📤 Upload Feedback", "🔬 Aspect Analysis", 
                        "🎯 Active Learning", "📊 Insights", "📈 Visualization", 
                        "👨‍💼 Admin Panel", "👤 Profile"],
                       label_visibility="collapsed")
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Show create workspace dialog
    if st.session_state.get('show_create_workspace', False):
        show_create_workspace_dialog()
    
    if page == "🏠 Home":
        show_home()
    elif page == "📤 Upload Feedback":
        show_upload()
    elif page == "🔬 Aspect Analysis":
        show_aspect_analysis()
    elif page == "🎯 Active Learning":
        show_active_learning()
    elif page == "📊 Insights":
        show_insights()
    elif page == "📈 Visualization":
        show_visualization()
    elif page == "👨‍💼 Admin Panel":
        show_admin_panel()
    elif page == "👤 Profile":
        show_profile()

def show_create_workspace_dialog():
    """Dialog to create new workspace"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📁 Create New Workspace")
    
    workspace_name = st.text_input("Workspace Name:", key="new_workspace_name")
    workspace_desc = st.text_area("Description:", key="new_workspace_desc")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Create", use_container_width=True):
            if workspace_name:
                create_workspace(st.session_state.user_id, workspace_name, workspace_desc)
                st.success(f"✅ Workspace '{workspace_name}' created!")
                st.session_state.show_create_workspace = False
                st.rerun()
            else:
                st.warning("⚠️ Please enter a workspace name")
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_create_workspace = False
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📂 Train Local CSV Files")
    csv_configs = [
        ("Product Reviews Dataset", "product_reviews.csv", "Product Reviews Workspace"),
        ("Emoji Reviews Dataset", "emoji_reviews.csv", "Emoji Reviews Workspace")
    ]
    
    reset_local = st.checkbox("Replace existing workspace data before import", value=True, key="reset_local_csvs")
    
    for label, file_name, workspace_name in csv_configs:
        file_path = Path(file_name)
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            if st.button(f"📥 Train {label}", key=f"train_{file_name}"):
                if not file_path.exists():
                    st.error(f"File not found: {file_path}")
                else:
                    with st.spinner(f"Importing and training on {label}..."):
                        result = bulk_train_from_csv(
                            str(file_path),
                            workspace_name,
                            st.session_state.user_id,
                            reset_workspace=reset_local
                        )
                    if result.get("trained"):
                        st.success(f"✅ {label} trained (v{result['version']}) | Acc: {result['accuracy']:.2f} | F1: {result['f1']:.2f}")
                    else:
                        st.warning(result.get("message", "Training failed"))
        with col_info:
            st.caption(f"{workspace_name} • Source: `{file_name}`")
    
    st.markdown("</div>", unsafe_allow_html=True)

def create_workspace(user_id, name, description=""):
    """Create a new workspace"""
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO workspaces (name, description, created_by, created_at) VALUES (?, ?, ?, ?)",
              (name, description, user_id, current_timestamp()))
    workspace_id = c.lastrowid
    conn.commit()
    conn.close()
    
    # Log activity
    log_activity(user_id, workspace_id, "create", f"Created workspace: {name}")
    
    return workspace_id


def get_or_create_workspace_by_name(user_id, name, description="Auto-generated workspace"):
    """Return workspace id for name, creating it if needed."""
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id FROM workspaces WHERE name=? LIMIT 1", (name,))
    row = c.fetchone()
    created = False
    if row:
        workspace_id = row[0]
    else:
        c.execute(
            "INSERT INTO workspaces (name, description, created_by, created_at) VALUES (?, ?, ?, ?)",
            (name, description, user_id, current_timestamp()))
        workspace_id = c.lastrowid
        created = True
    conn.commit()
    conn.close()
    if created:
        log_activity(user_id, workspace_id, "create", f"Auto-created workspace: {name}")
    return workspace_id

def show_profile():
    st.markdown("<h1 style='text-align: center;'>🏠 Dashboard Overview</h1>", unsafe_allow_html=True)
    
    df = get_user_feedback(st.session_state.user_id)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Total Feedback", len(df))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        positive = len(df[df['sentiment'] == 'Positive']) if len(df) > 0 else 0
        st.metric("Positive", positive)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        neutral = len(df[df['sentiment'] == 'Neutral']) if len(df) > 0 else 0
        st.metric("Neutral", neutral)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        negative = len(df[df['sentiment'] == 'Negative']) if len(df) > 0 else 0
        st.metric("Negative", negative)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if len(df) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📝 Recent Feedback")
        recent_df = df.head(5)[['feedback_text', 'sentiment', 'confidence_score', 'created_at']].copy()
        if 'confidence_score' in recent_df.columns:
            recent_df['confidence_score'] = recent_df['confidence_score'].apply(lambda x: f"{x:.2%}")
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

def show_upload():
    st.markdown("<h1 style='text-align: center;'>📤 Upload & Analyze Feedback</h1>", unsafe_allow_html=True)
    
    # AI Settings Panel
    with st.expander("⚙️ AI Analysis Settings", expanded=False):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            ai_enabled = st.toggle("🤖 Enable AI-Powered Analysis", value=USE_AI_ANALYSIS, 
                                  help="Uses Claude AI for more accurate aspect sentiment detection",
                                  key="ai_toggle")
            if ai_enabled:
                st.success("✅ AI Analysis Enabled - Higher accuracy!")
            else:
                st.info("ℹ️ Using rule-based analysis")
        with col_s2:
            st.info("**AI Benefits:**\n- Better context understanding\n- More accurate sentiment detection\n- Handles complex sentences\n- Explains reasoning")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📄 Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_upload)} feedback entries!")
                st.dataframe(df_upload.head(), use_container_width=True)
                
                text_column = st.selectbox("Select feedback text column:", df_upload.columns)
                
                if st.button("🚀 Analyze All Feedback", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Get AI setting
                    use_ai = st.session_state.get('ai_toggle', USE_AI_ANALYSIS)
                    
                    print("\n" + "="*80)
                    print(f"🔍 {'AI-POWERED' if use_ai else 'RULE-BASED'} BATCH FEEDBACK ANALYSIS STARTED")
                    print("="*80)
                    
                    for idx, row in df_upload.iterrows():
                        status_text.text(f"{'🤖 AI-Analyzing' if use_ai else 'Analyzing'} feedback {idx+1}/{len(df_upload)}...")
                        progress_bar.progress((idx + 1) / len(df_upload))
                        
                        text = str(row[text_column])
                        cleaned_text = preprocess_text(text)
                        sentiment, score, confidence, raw_conf = get_sentiment_prediction(
                            text, st.session_state.get('current_workspace'))
                        aspects = perform_absa(text, use_ai=use_ai)
                        predicted_intent, predicted_entities = predict_intent_and_entities(text)
                        
                        # Print detailed analysis
                        print(f"\n📝 Feedback #{idx+1}:")
                        print(f"Text: {text}")
                        print(f"Overall Sentiment: {sentiment} (Score: {score:.3f}, Confidence: {confidence:.2%})")
                        
                        if aspects:
                            print(f"🎯 {'AI-Detected' if use_ai else 'Detected'} Aspects ({len(aspects)} found):")
                            for asp in aspects:
                                print(f"  • {asp['aspect'].upper()}")
                                print(f"    Keywords: {', '.join(asp['keywords'])}")
                                print(f"    Sentiment: {asp['sentiment']}")
                                print(f"    Score: {asp['score']:.3f}")
                                print(f"    Confidence: {asp['confidence']:.2%}")
                                if 'reasoning' in asp:
                                    print(f"    AI Reasoning: {asp['reasoning']}")
                        else:
                            print("  ⚠️ No specific aspects detected")
                        print("-" * 80)
                        
                        save_feedback_with_aspects(
                            st.session_state.user_id, text, cleaned_text,
                            sentiment, score, confidence, aspects,
                            st.session_state.get('current_workspace'),
                            predicted_intent=predicted_intent,
                            predicted_entities=predicted_entities,
                            raw_confidence=raw_conf
                        )
                    
                    print(f"\n✅ {'AI-POWERED' if use_ai else 'RULE-BASED'} BATCH ANALYSIS COMPLETED!")
                    print("="*80 + "\n")
                    
                    # Log activity
                    log_activity(
                        st.session_state.user_id,
                        st.session_state.get('current_workspace'),
                        "upload",
                        f"Uploaded {len(df_upload)} feedback entries"
                    )
                    
                    status_text.text("")
                    progress_bar.empty()
                    st.success(f"✅ {'AI-Analyzed' if use_ai else 'Analyzed'} and saved {len(df_upload)} feedback entries!")
                    st.balloons()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                print(f"❌ ERROR: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ✍️ Manual Text Input")
        feedback_text = st.text_area("Enter customer feedback:", height=200,
            placeholder="Example: The battery life is amazing and the camera quality is excellent, but the screen is a bit dim in sunlight.")
        
        if st.button("🔍 Analyze Feedback", use_container_width=True):
            if feedback_text:
                # Get AI setting
                use_ai = st.session_state.get('ai_toggle', USE_AI_ANALYSIS)
                
                with st.spinner(f"🔮 Analyzing with {'AI-Powered' if use_ai else 'Advanced'} ABSA..."):
                    cleaned_text = preprocess_text(feedback_text)
                    sentiment, score, confidence, raw_conf = get_sentiment_prediction(
                        feedback_text, st.session_state.get('current_workspace'))
                    
                    # Show AI analysis status
                    ai_status = st.empty()
                    if use_ai:
                        ai_status.info("🤖 Using Advanced AI Analysis for higher accuracy...")
                    
                    aspects = perform_absa(feedback_text, use_ai=use_ai)
                    ai_status.empty()
                    
                    # Print detailed analysis to terminal
                    print("\n" + "="*80)
                    print(f"🔍 {'AI-POWERED' if use_ai else 'ADVANCED'} FEEDBACK ANALYSIS")
                    print("="*80)
                    print(f"\n📝 Original Text:\n{feedback_text}")
                    print(f"\n🧹 Cleaned Text:\n{cleaned_text}")
                    print(f"\n💭 Overall Sentiment: {sentiment}")
                    print(f"   Score: {score:.3f}")
                    print(f"   Confidence: {confidence:.2%}")
                    
                    if aspects:
                        print(f"\n🎯 {'AI-Extracted' if use_ai else 'Extracted'} Aspects & Sentiments ({len(aspects)} found):")
                        print("-" * 80)
                        for i, asp in enumerate(aspects, 1):
                            print(f"{i}. Aspect: '{asp['aspect'].upper()}'")
                            print(f"   Matched Keywords: {', '.join(asp['keywords'])}")
                            print(f"   Sentiment: {asp['sentiment']}")
                            print(f"   Score: {asp['score']:.3f}")
                            print(f"   Confidence: {asp['confidence']:.2%}")
                            if 'reasoning' in asp:
                                print(f"   AI Reasoning: {asp['reasoning']}")
                            print()
                    else:
                        print("\n⚠️  No specific aspects detected in this feedback")
                    
                    print("="*80 + "\n")
                    
                    predicted_intent, predicted_entities = predict_intent_and_entities(feedback_text)
                    save_feedback_with_aspects(
                        st.session_state.user_id, feedback_text, cleaned_text,
                        sentiment, score, confidence, aspects,
                        st.session_state.get('current_workspace'),
                        predicted_intent=predicted_intent,
                        predicted_entities=predicted_entities,
                        raw_confidence=raw_conf
                    )
                    
                    # Log activity
                    log_activity(
                        st.session_state.user_id,
                        st.session_state.get('current_workspace'),
                        "upload",
                        "Uploaded single feedback entry"
                    )
                    
                    st.success("✅ Feedback analyzed and saved!")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Overall Sentiment", sentiment)
                    with col_b:
                        st.metric("Score", f"{score:.3f}")
                    with col_c:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    if aspects:
                        st.markdown("---")
                        st.markdown(f"### 🎯 {'AI-Detected' if use_ai else 'Detected'} Aspects:")
                        for asp in aspects:
                            sentiment_class = f"aspect-{asp['sentiment'].lower()}"
                            reasoning_text = f"<br><span style='color: #00F0FF; font-size: 0.85em; font-style: italic;'>💡 AI: {asp['reasoning']}</span>" if 'reasoning' in asp else ""
                            st.markdown(f"""
                            <div style='margin: 10px 0; padding: 10px; border-left: 3px solid #B19CD9; background: rgba(45, 27, 78, 0.3); border-radius: 10px;'>
                                <span class='{sentiment_class}' style='font-size: 1.1em;'>
                                    <strong>{asp['aspect'].title()}</strong>: {asp['sentiment']}
                                </span>
                                <br>
                                <span style='color: #B19CD9; font-size: 0.9em;'>
                                    Keywords: {', '.join(asp['keywords'])} | Score: {asp['score']:.3f} | Confidence: {asp['confidence']:.2%}
                                </span>
                                {reasoning_text}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("---")
                        if use_ai:
                            st.info("ℹ️ **AI Analysis:** This feedback is too general and doesn't mention specific product aspects to analyze.")
                        else:
                            st.info("ℹ️ **No specific product aspects detected.** Enable AI Analysis for smarter aspect detection.")
            else:
                st.warning("⚠️ Please enter feedback text!")
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_aspect_analysis():
    """AI-Powered Aspect-Based Analysis Page with Pagination"""
    st.markdown("<h1 style='text-align: center;'>🔬 AI-Powered Aspect Analysis</h1>", unsafe_allow_html=True)
    
    workspace_id = st.session_state.get('current_workspace')
    if not workspace_id:
        st.warning("⚠️ Please select a workspace first!")
        return
    
    df = get_user_feedback(st.session_state.user_id, workspace_id)
    
    if len(df) == 0:
        st.info("📭 No feedback data available. Upload some feedback first!")
        return
    
    # Tab view
    tab1, tab2, tab3 = st.tabs(["📋 Review with Aspects", "📊 Aspect Breakdown", "🔍 Aspect Search"])
    
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📝 All Reviews with Advanced Aspect Extraction")
        
        # Add pagination controls
        items_per_page = st.selectbox("Reviews per page:", [10, 25, 50, 100], index=1, key="reviews_per_page")
        total_pages = (len(df) + items_per_page - 1) // items_per_page
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        
        with col_prev:
            if st.button("⬅️ Previous", disabled=(st.session_state.current_page == 1), key="prev_reviews"):
                st.session_state.current_page -= 1
                st.rerun()
        
        with col_info:
            st.markdown(f"<p style='text-align: center; color: #B19CD9; font-size: 1.1em;'>Page {st.session_state.current_page} of {total_pages} | Total Reviews: {len(df)}</p>", unsafe_allow_html=True)
        
        with col_next:
            if st.button("Next ➡️", disabled=(st.session_state.current_page == total_pages), key="next_reviews"):
                st.session_state.current_page += 1
                st.rerun()
        
        st.markdown("---")
        
        # Calculate start and end indices
        start_idx = (st.session_state.current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(df))
        
        # Display reviews for current page
        for idx, row in df.iloc[start_idx:end_idx].iterrows():
            with st.expander(f"📝 Review #{start_idx + df.iloc[start_idx:end_idx].index.get_loc(idx) + 1}: {row['feedback_text'][:80]}..."):
                st.markdown(f"**Full Review:** {row['feedback_text']}")
                st.markdown(f"**Overall Sentiment:** {row['sentiment']} (Score: {row['sentiment_score']:.3f}, Confidence: {row['confidence_score']:.2%})")
                st.markdown(f"**Date:** {row['created_at']}")
                
                if 'aspects' in row and row['aspects']:
                    try:
                        aspects = json.loads(row['aspects'])
                        if aspects:
                            st.markdown("---")
                            st.markdown(f"**AI-Detected Aspects: {len(aspects)}**")
                            
                            for asp in aspects:
                                sentiment_class = f"aspect-{asp['sentiment'].lower()}"
                                keywords_str = ', '.join(asp.get('keywords', [asp['aspect']]))
                                reasoning_html = f"<br><span style='color: #00F0FF; font-size: 0.85em; font-style: italic;'>💡 AI: {asp['reasoning']}</span>" if 'reasoning' in asp else ""
                                st.markdown(f"""
                                <div style='margin: 10px 0; padding: 12px; border-left: 4px solid #B19CD9; background: rgba(45, 27, 78, 0.4); border-radius: 10px;'>
                                    <span class='{sentiment_class}' style='font-size: 1.05em;'>
                                        🎯 <strong>{asp['aspect'].title()}</strong>: {asp['sentiment']}
                                    </span>
                                    <br>
                                    <span style='color: #E6E6FA; font-size: 0.9em;'>
                                        Keywords: <em>{keywords_str}</em>
                                    </span>
                                    <br>
                                    <span style='color: #B19CD9; font-size: 0.85em;'>
                                        Score: {asp['score']:.3f} | Confidence: {asp['confidence']:.2%}
                                    </span>
                                    {reasoning_html}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No specific aspects detected")
                    except Exception as e:
                        st.info(f"Aspect data not available")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Comprehensive Aspect Sentiment Breakdown")
        
        # Aggregate aspect data with advanced metrics
        all_aspects = defaultdict(lambda: {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'scores': [],
            'confidences': [],
            'keywords': set()
        })
        
        for idx, row in df.iterrows():
            if 'aspects' in row and row['aspects']:
                try:
                    aspects = json.loads(row['aspects'])
                    for asp in aspects:
                        aspect_name = asp['aspect']
                        sentiment = asp['sentiment'].lower()
                        
                        all_aspects[aspect_name][sentiment] += 1
                        all_aspects[aspect_name]['scores'].append(asp['score'])
                        all_aspects[aspect_name]['confidences'].append(asp['confidence'])
                        if 'keywords' in asp:
                            all_aspects[aspect_name]['keywords'].update(asp['keywords'])
                        else:
                            all_aspects[aspect_name]['keywords'].add(aspect_name)
                except:
                    pass
        
        if all_aspects:
            # Create comprehensive summary table
            aspect_summary = []
            for aspect, data in all_aspects.items():
                total = data['positive'] + data['negative'] + data['neutral']
                avg_score = np.mean(data['scores']) if data['scores'] else 0
                avg_confidence = np.mean(data['confidences']) if data['confidences'] else 0
                pos_pct = (data['positive'] / total * 100) if total > 0 else 0
                neg_pct = (data['negative'] / total * 100) if total > 0 else 0
                
                aspect_summary.append({
                    'Aspect': aspect.title(),
                    'Total Mentions': total,
                    'Positive': data['positive'],
                    'Negative': data['negative'],
                    'Neutral': data['neutral'],
                    'Positive %': f"{pos_pct:.1f}%",
                    'Negative %': f"{neg_pct:.1f}%",
                    'Avg Score': f"{avg_score:.3f}",
                    'Avg Confidence': f"{avg_confidence:.2%}",
                    'Keywords': ', '.join(list(data['keywords'])[:5])
                })
            
            summary_df = pd.DataFrame(aspect_summary).sort_values('Total Mentions', ascending=False)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Visualization
            st.markdown("---")
            st.markdown("### 📈 Aspect Sentiment Visualization")
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Grouped bar chart
                top_aspects = summary_df.head(10)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Positive', x=top_aspects['Aspect'], y=top_aspects['Positive'],
                                    marker_color='#00F0FF'))
                fig.add_trace(go.Bar(name='Negative', x=top_aspects['Aspect'], y=top_aspects['Negative'],
                                    marker_color='#FF6EC7'))
                fig.add_trace(go.Bar(name='Neutral', x=top_aspects['Aspect'], y=top_aspects['Neutral'],
                                    marker_color='#B19CD9'))
                
                fig.update_layout(
                    title="Top 10 Aspects by Sentiment Count",
                    barmode='group',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E6E6FA'),
                    xaxis=dict(title="Aspect", tickangle=-45),
                    yaxis=dict(title="Count"),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col_viz2:
                # Sentiment distribution pie chart
                total_pos = sum(data['positive'] for data in all_aspects.values())
                total_neg = sum(data['negative'] for data in all_aspects.values())
                total_neu = sum(data['neutral'] for data in all_aspects.values())
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Positive', 'Negative', 'Neutral'],
                    values=[total_pos, total_neg, total_neu],
                    hole=0.4,
                    marker=dict(colors=['#00F0FF', '#FF6EC7', '#B19CD9']),
                    textfont=dict(size=16, color='white')
                )])
                
                fig.update_layout(
                    title="Overall Aspect Sentiment Distribution",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E6E6FA', size=14),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No aspect data available yet. Analyze some feedback with aspects!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Search and Filter by Aspect")
        
        # Get all unique aspects
        all_aspect_names = set()
        for idx, row in df.iterrows():
            if 'aspects' in row and row['aspects']:
                try:
                    aspects = json.loads(row['aspects'])
                    for asp in aspects:
                        all_aspect_names.add(asp['aspect'])
                except:
                    pass
        
        if all_aspect_names:
            selected_aspect = st.selectbox("Select an aspect to analyze:", sorted(all_aspect_names), key="aspect_search")
            
            sentiment_filter = st.multiselect("Filter by sentiment:", ["Positive", "Negative", "Neutral"], 
                                             default=["Positive", "Negative", "Neutral"])
            
            if selected_aspect:
                st.markdown(f"### Analysis for: **{selected_aspect.title()}**")
                
                # Filter feedback containing this aspect
                matching_feedback = []
                for idx, row in df.iterrows():
                    if 'aspects' in row and row['aspects']:
                        try:
                            aspects = json.loads(row['aspects'])
                            for asp in aspects:
                                if asp['aspect'] == selected_aspect and asp['sentiment'] in sentiment_filter:
                                    matching_feedback.append({
                                        'text': row['feedback_text'],
                                        'sentiment': asp['sentiment'],
                                        'score': asp['score'],
                                        'confidence': asp['confidence'],
                                        'keywords': asp.get('keywords', [selected_aspect]),
                                        'reasoning': asp.get('reasoning', ''),
                                        'date': row['created_at']
                                    })
                        except:
                            pass
                
                if matching_feedback:
                    st.metric("Total Mentions (Filtered)", len(matching_feedback))
                    
                    # Sentiment breakdown
                    sentiment_counts = Counter([f['sentiment'] for f in matching_feedback])
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Positive", sentiment_counts.get('Positive', 0))
                    with col2:
                        st.metric("Neutral", sentiment_counts.get('Neutral', 0))
                    with col3:
                        st.metric("Negative", sentiment_counts.get('Negative', 0))
                    with col4:
                        avg_score = np.mean([f['score'] for f in matching_feedback])
                        st.metric("Avg Score", f"{avg_score:.3f}")
                    
                    st.markdown("---")
                    st.markdown("### 📝 Reviews mentioning this aspect:")
                    
                    # Pagination for search results
                    results_per_page = 10
                    total_result_pages = (len(matching_feedback) + results_per_page - 1) // results_per_page
                    
                    if 'search_page' not in st.session_state:
                        st.session_state.search_page = 1
                    
                    col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
                    
                    with col_p1:
                        if st.button("⬅️ Previous", disabled=(st.session_state.search_page == 1), key="prev_search"):
                            st.session_state.search_page -= 1
                            st.rerun()
                    
                    with col_p2:
                        st.markdown(f"<p style='text-align: center; color: #B19CD9;'>Page {st.session_state.search_page} of {total_result_pages}</p>", unsafe_allow_html=True)
                    
                    with col_p3:
                        if st.button("Next ➡️", disabled=(st.session_state.search_page == total_result_pages), key="next_search"):
                            st.session_state.search_page += 1
                            st.rerun()
                    
                    start_res = (st.session_state.search_page - 1) * results_per_page
                    end_res = min(start_res + results_per_page, len(matching_feedback))
                    
                    for feedback in matching_feedback[start_res:end_res]:
                        sentiment_class = f"aspect-{feedback['sentiment'].lower()}"
                        keywords_str = ', '.join(feedback['keywords'])
                        reasoning_html = f"<br><span style='color: #00F0FF; font-size: 0.85em; font-style: italic;'>💡 AI: {feedback['reasoning']}</span>" if feedback.get('reasoning') else ""
                        st.markdown(f"""
                        <div style='margin: 15px 0; padding: 15px; border-left: 4px solid #B19CD9; background: rgba(45, 27, 78, 0.3); border-radius: 10px;'>
                            <p style='color: #E6E6FA; font-size: 1.05em;'>{feedback['text']}</p>
                            <span class='{sentiment_class}' style='font-size: 1.1em;'>
                                <strong>{feedback['sentiment']}</strong>
                            </span>
                            <span style='color: #B19CD9; font-size: 0.9em;'>
                                 | Score: {feedback['score']:.3f} | Confidence: {feedback['confidence']:.2%}
                            </span>
                            <br>
                            <span style='color: #B19CD9; font-size: 0.85em;'>
                                Keywords: {keywords_str}
                            </span>
                            {reasoning_html}
                            <p style='color: #B19CD9; font-size: 0.85em; margin-top: 8px;'>📅 {feedback['date'][:19]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No feedback found for this aspect with the selected sentiment filters")
        else:
            st.info("No aspects available. Upload and analyze feedback with aspect extraction!")
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_insights():
    st.markdown("<h1 style='text-align: center;'>📊 Advanced Insights</h1>", unsafe_allow_html=True)
    
    workspace_id = st.session_state.get('current_workspace')
    if not workspace_id:
        st.warning("⚠️ Please select a workspace first!")
        return
    
    df = get_user_feedback(st.session_state.user_id, workspace_id)
    
    if len(df) == 0:
        st.info("📭 No feedback data available!")
        return
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎭 Overall Sentiment Distribution")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        sentiment_counts = df['sentiment'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,
            marker=dict(colors=['#00F0FF', '#B19CD9', '#FF6EC7']),
            textfont=dict(size=16, color='white')
        )])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E6E6FA', size=14),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker=dict(color=['#00F0FF', '#B19CD9', '#FF6EC7']),
            text=sentiment_counts.values,
            textposition='outside'
        )])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E6E6FA'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_visualization():
    st.markdown("<h1 style='text-align: center;'>📈 Data Visualization</h1>", unsafe_allow_html=True)
    
    workspace_id = st.session_state.get('current_workspace')
    if not workspace_id:
        st.warning("⚠️ Please select a workspace first!")
        return
    
    df = get_user_feedback(st.session_state.user_id, workspace_id)
    
    if len(df) == 0:
        st.info("📭 No feedback data available!")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Sentiment Over Time")
        
        df['created_at'] = pd.to_datetime(df['created_at'])
        df_sorted = df.sort_values('created_at')
        
        fig = px.scatter(df_sorted, x='created_at', y='sentiment_score',
                        color='sentiment',
                        color_discrete_map={
                            'Positive': '#00F0FF',
                            'Neutral': '#B19CD9',
                            'Negative': '#FF6EC7'
                        })
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E6E6FA'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ☁️ Word Cloud")
        
        all_text = ' '.join(df['cleaned_text'].dropna().apply(str))
        
        if all_text:
            fig = create_wordcloud(all_text)
            if fig:
                st.pyplot(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_active_learning():
    """Active Learning Module - Review uncertain predictions"""
    st.markdown("<h1 style='text-align: center;'>🎓 Active Learning - Review Uncertain Predictions</h1>", unsafe_allow_html=True)
    
    if 'current_workspace' not in st.session_state or not st.session_state.current_workspace:
        st.warning("⚠️ Please select a workspace first!")
        return
    
    workspace_id = st.session_state.current_workspace
    
    st.markdown("""
    <div class='card'>
    <h3>📚 About Active Learning</h3>
    <p>Active Learning helps improve model accuracy by:</p>
    <ul>
        <li>⏱️ Saving time - focuses only on uncertain predictions</li>
        <li>🎯 Improving model quality faster</li>
        <li>📉 Reducing manual annotation workload</li>
        <li>🔍 Focusing only on problematic data</li>
    </ul>
    <p>Review predictions with confidence below <strong>{:.0f}%</strong> threshold.</p>
    </div>
    """.format(CONFIDENCE_THRESHOLD * 100), unsafe_allow_html=True)
    
    uncertain_df = get_uncertain_feedback(st.session_state.user_id, workspace_id)
    
    if len(uncertain_df) == 0:
        st.success("🎉 No uncertain predictions! All feedback has been analyzed with high confidence.")
        return
    
    st.info(f"📊 **{len(uncertain_df)} predictions** need your review (confidence < {CONFIDENCE_THRESHOLD*100:.0f}%)")
    
    # Add pagination for uncertain feedback
    items_per_page = 5
    total_pages = (len(uncertain_df) + items_per_page - 1) // items_per_page
    
    if 'active_learning_page' not in st.session_state:
        st.session_state.active_learning_page = 1
    
    col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
    
    with col_p1:
        if st.button("⬅️ Previous", disabled=(st.session_state.active_learning_page == 1), key="prev_al"):
            st.session_state.active_learning_page -= 1
            st.rerun()
    
    with col_p2:
        st.markdown(f"<p style='text-align: center; color: #B19CD9; font-size: 1.1em;'>Page {st.session_state.active_learning_page} of {total_pages}</p>", unsafe_allow_html=True)
    
    with col_p3:
        if st.button("Next ➡️", disabled=(st.session_state.active_learning_page == total_pages), key="next_al"):
            st.session_state.active_learning_page += 1
            st.rerun()
    
    st.markdown("---")
    
    start_idx = (st.session_state.active_learning_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(uncertain_df))
    
    for idx, row in uncertain_df.iloc[start_idx:end_idx].iterrows():
        with st.container():
            st.markdown("<div class='uncertain-box'>", unsafe_allow_html=True)
            
            row_idx = start_idx + list(uncertain_df.iloc[start_idx:end_idx].index).index(idx) + 1
            st.markdown(f"### 📝 Feedback #{row_idx}")
            st.markdown(f"**Text:** {row['feedback_text']}")
            st.markdown(f"**Date:** {row.get('created_at', 'N/A')}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                predicted_sentiment = row.get('predicted_sentiment') or row.get('sentiment', 'N/A')
                st.metric("Predicted Sentiment", predicted_sentiment)
            with col2:
                st.metric("Score", f"{row.get('sentiment_score', 0):.3f}")
            with col3:
                conf_score = row.get('confidence_score', 0) or 0
                st.metric("⚠️ Confidence", f"{conf_score:.2%}", 
                         delta=f"{(conf_score-CONFIDENCE_THRESHOLD)*100:.1f}%")
            
            # Show predicted aspects
            if row.get('predicted_aspects') or row.get('aspects'):
                try:
                    aspects_data = row.get('predicted_aspects') or row.get('aspects')
                    aspects = json.loads(aspects_data) if aspects_data else []
                    if aspects:
                        st.markdown("**Predicted Aspects:**")
                        for asp in aspects:
                            conf = asp.get('confidence', 0) or 0
                            st.markdown(f"- {asp['aspect'].title()}: {asp['sentiment']} (Conf: {conf:.2%})")
                except:
                    pass
            
            st.markdown("---")
            st.markdown("### ✏️ Provide Corrections")
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                predicted_sentiment = row.get('predicted_sentiment') or row.get('sentiment', 'Neutral')
                sentiment_options = ["Positive", "Neutral", "Negative"]
                try:
                    default_idx = sentiment_options.index(predicted_sentiment)
                except:
                    default_idx = 1
                corrected_sentiment = st.selectbox(
                    "Correct Sentiment",
                    sentiment_options,
                    index=default_idx,
                    key=f"sent_{row['id']}"
                )
            
            with col_c2:
                # Allow editing aspects
                st.markdown("**Correct Aspects (optional):**")
                available_aspects = list(ASPECT_CATEGORIES.keys())
                
                try:
                    aspects_data = row.get('predicted_aspects') or row.get('aspects')
                    predicted_aspects = json.loads(aspects_data) if aspects_data else []
                    default_aspects = [asp['aspect'] for asp in predicted_aspects]
                except:
                    default_aspects = []
                
                corrected_aspect_names = st.multiselect(
                    "Select aspects",
                    available_aspects,
                    default=default_aspects,
                    key=f"asp_{row['id']}"
                )
            
            feedback_remarks = st.text_area(
                "Feedback Remarks (optional)",
                placeholder="Add any notes about this correction...",
                key=f"remarks_{row['id']}",
                height=80
            )
            
            if st.button("💾 Save Correction", key=f"save_{row['id']}", use_container_width=True):
                corrected_aspects = [{'aspect': asp, 'sentiment': corrected_sentiment} 
                                    for asp in corrected_aspect_names]
                
                save_feedback_correction(
                    row['id'], 
                    st.session_state.user_id,
                    corrected_sentiment,
                    corrected_aspects,
                    feedback_remarks
                )
                
                st.success("✅ Correction saved! This will help improve the model.")
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

def show_active_learning_debug():
    """Debug version to see what's happening"""
    st.markdown("<h1 style='text-align: center;'>🎯 Active Learning - DEBUG MODE</h1>", unsafe_allow_html=True)
    
    # Check session state
    st.markdown("### 🔍 Debug Information")
    st.markdown(f"**Logged in user ID:** {st.session_state.get('user_id', 'NOT SET')}")
    st.markdown(f"**Current workspace ID:** {st.session_state.get('current_workspace', 'NOT SET')}")
    
    workspace_id = st.session_state.get('current_workspace')
    
    if not workspace_id:
        st.error("❌ No workspace selected!")
        return
    
    # Check database directly
    conn = sqlite3.connect('users.db', check_same_thread=False)
    
    # Check if feedback exists at all
    st.markdown("---")
    st.markdown("### 📊 Database Check")
    
    total_feedback = pd.read_sql_query("SELECT COUNT(*) as count FROM feedback", conn)
    st.markdown(f"**Total feedback in database:** {total_feedback['count'].iloc[0]}")
    
    # Check feedback for this workspace
    workspace_feedback = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM feedback WHERE workspace_id = ?", 
        conn, params=(workspace_id,))
    st.markdown(f"**Feedback for current workspace ({workspace_id}):** {workspace_feedback['count'].iloc[0]}")
    
    # Show all workspaces
    st.markdown("---")
    st.markdown("### 📁 All Workspaces")
    workspaces = pd.read_sql_query("SELECT * FROM workspaces", conn)
    st.dataframe(workspaces, use_container_width=True)
    
    # Show feedback with workspace info
    st.markdown("---")
    st.markdown("### 📝 All Feedback with Workspace Info")
    all_feedback = pd.read_sql_query("""
        SELECT f.id, f.workspace_id, w.name as workspace_name, 
               f.feedback_text, f.sentiment, f.confidence_score, f.raw_confidence
        FROM feedback f
        LEFT JOIN workspaces w ON f.workspace_id = w.id
        ORDER BY f.created_at DESC
        LIMIT 20
    """, conn)
    st.dataframe(all_feedback, use_container_width=True)
    
    # Show feedback for selected workspace specifically
    st.markdown("---")
    st.markdown(f"### 📝 Feedback for Workspace ID: {workspace_id}")
    selected_workspace_feedback = pd.read_sql_query("""
        SELECT f.id, f.feedback_text, f.sentiment, f.confidence_score, f.raw_confidence, f.created_at
        FROM feedback f
        WHERE f.workspace_id = ?
        ORDER BY f.created_at DESC
    """, conn, params=(workspace_id,))
    
    if len(selected_workspace_feedback) > 0:
        st.success(f"✅ Found {len(selected_workspace_feedback)} feedback entries!")
        st.dataframe(selected_workspace_feedback, use_container_width=True)
        
        # Show confidence statistics
        st.markdown("---")
        st.markdown("### 📊 Confidence Statistics")
        st.markdown(f"**Min Confidence:** {selected_workspace_feedback['confidence_score'].min():.2%}")
        st.markdown(f"**Max Confidence:** {selected_workspace_feedback['confidence_score'].max():.2%}")
        st.markdown(f"**Avg Confidence:** {selected_workspace_feedback['confidence_score'].mean():.2%}")
        
        # Test the filter
        threshold = st.slider("Test threshold:", 0.0, 1.0, 0.70, 0.05)
        filtered = selected_workspace_feedback[selected_workspace_feedback['confidence_score'] < threshold]
        st.markdown(f"**Entries below {threshold:.0%}:** {len(filtered)}")
        
    else:
        st.error("❌ No feedback found for this workspace!")
    
    conn.close()
def save_feedback_correction(feedback_id, user_id, corrected_sentiment, corrected_aspects, feedback_remarks=""):
    """Save user corrections for uncertain predictions"""
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    # Get workspace_id for logging
    c.execute("SELECT workspace_id FROM feedback WHERE id=?", (feedback_id,))
    workspace_result = c.fetchone()
    workspace_id = workspace_result[0] if workspace_result else None
    
    c.execute("""UPDATE feedback 
                 SET corrected_sentiment=?, corrected_aspects=?, 
                     is_corrected=1, needs_review=0 
                 WHERE id=? AND user_id=?""",
              (corrected_sentiment, json.dumps(corrected_aspects), feedback_id, user_id))
    conn.commit()
    conn.close()
    
    # Also save to corrections table for history
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    # Get original values
    c.execute("SELECT sentiment, predicted_sentiment, aspects, predicted_aspects FROM feedback WHERE id=?", (feedback_id,))
    orig = c.fetchone()
    if orig:
        original_sentiment = orig[0] or orig[1]
        original_aspects = orig[2] or orig[3]
        c.execute("""INSERT INTO corrections 
                     (feedback_id, user_id, original_sentiment, corrected_sentiment, 
                      corrected_aspects, remarks) 
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  (feedback_id, user_id, original_sentiment, corrected_sentiment,
                   json.dumps(corrected_aspects), feedback_remarks))
        conn.commit()
    conn.close()
    
    # Log the correction activity
    if workspace_id:
        log_activity(
            user_id,
            workspace_id,
            "feedback_correction",
            f"Corrected feedback ID {feedback_id}: {corrected_sentiment}"
        )

def save_correction(
        feedback_id,
        user_id,
        original_sentiment,
        corrected_sentiment,
        corrected_aspects,
        remarks,
        original_intent=None,
        corrected_intent=None,
        original_entities=None,
        corrected_entities=None):
    """Save user correction to database (legacy function for compatibility)"""
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("""INSERT INTO corrections 
                 (feedback_id, user_id, original_sentiment, corrected_sentiment, 
                  corrected_aspects, remarks, original_intent, corrected_intent,
                  original_entities, corrected_entities) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (feedback_id, user_id, original_sentiment, corrected_sentiment,
               corrected_aspects, remarks, original_intent, corrected_intent,
               original_entities, corrected_entities))
    
    c.execute("""UPDATE feedback 
                 SET sentiment=?, corrected_intent=?, corrected_entities=?, raw_confidence=1.0,
                     is_corrected=1, needs_review=0
                 WHERE id=?""",
              (corrected_sentiment, corrected_intent, corrected_entities, feedback_id))
    conn.commit()
    conn.close()

def show_admin_panel():
    """Admin Panel - Only accessible by admin users"""
    st.markdown("<h1 style='text-align: center;'>👨‍💼 Admin Panel</h1>", unsafe_allow_html=True)
    
    # Check if user is admin (first registered user or role=admin)
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id, is_admin FROM users WHERE id=?", (st.session_state.user_id,))
    user = c.fetchone()
    conn.close()
    
    if not user or not user[1]:
        st.error("🚫 Access Denied: Admin privileges required")
        st.info("Only administrators can access this panel.")
        return
    
    # Admin tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 Users", "📁 Workspaces", "📊 Datasets", "🤖 Models", "📜 Activity Logs"
    ])
    
    with tab1:
        show_user_management()
    
    with tab2:
        show_workspace_management()
    
    with tab3:
        show_dataset_management()
    
    with tab4:
        show_model_management()
    
    with tab5:
        show_activity_logs()

def show_user_management():
    """User Management Section"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 👥 User Management")
    
    # Get all users
    conn = sqlite3.connect('users.db', check_same_thread=False)
    users_df = pd.read_sql_query("""
        SELECT u.id, u.username, u.email, u.is_admin, u.created_at,
               COUNT(DISTINCT w.id) as workspaces,
               COUNT(DISTINCT f.id) as feedback_count
        FROM users u
        LEFT JOIN workspaces w ON u.id = w.created_by
        LEFT JOIN feedback f ON w.id = f.workspace_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """, conn)
    conn.close()
    
    st.dataframe(users_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 🔧 User Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Remove User:**")
        user_to_remove = st.selectbox("Select user:", users_df['username'].tolist(), key="remove_user")
        
        if st.button("🗑️ Remove User", use_container_width=True):
            user_id = users_df[users_df['username'] == user_to_remove]['id'].values[0]
            if user_id == st.session_state.user_id:
                st.error("❌ Cannot remove yourself!")
            else:
                conn = sqlite3.connect('users.db', check_same_thread=False)
                c = conn.cursor()
                c.execute("DELETE FROM users WHERE id=?", (user_id,))
                conn.commit()
                conn.close()
                st.success(f"✅ User '{user_to_remove}' removed!")
                st.rerun()
    
    with col2:
        st.markdown("**Reset Password (Optional):**")
        user_to_reset = st.selectbox("Select user:", users_df['username'].tolist(), key="reset_user")
        new_password = st.text_input("New Password:", type="password", key="new_pwd")
        
        if st.button("🔄 Reset Password", use_container_width=True):
            if new_password and len(new_password) >= 6:
                user_id = users_df[users_df['username'] == user_to_reset]['id'].values[0]
                conn = sqlite3.connect('users.db', check_same_thread=False)
                c = conn.cursor()
                c.execute("UPDATE users SET password_hash=? WHERE id=?",
                         (hash_password(new_password), user_id))
                conn.commit()
                conn.close()
                st.success(f"✅ Password reset for '{user_to_reset}'!")
            else:
                st.warning("⚠️ Password must be at least 6 characters")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_workspace_management():
    """Workspace Management Section"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📁 Workspace Management")
    
    # Get all workspaces
    conn = sqlite3.connect('users.db', check_same_thread=False)
    workspaces_df = pd.read_sql_query("""
        SELECT w.id, w.name, w.description, u.username as created_by, w.created_at,
               COUNT(DISTINCT f.id) as feedback_count,
               COUNT(DISTINCT m.id) as model_count
        FROM workspaces w
        JOIN users u ON w.created_by = u.id
        LEFT JOIN feedback f ON w.id = f.workspace_id
        LEFT JOIN models m ON w.id = m.workspace_id
        GROUP BY w.id
        ORDER BY w.created_at DESC
    """, conn)
    conn.close()
    
    st.dataframe(workspaces_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 🔧 Workspace Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Delete Workspace:**")
        workspace_to_delete = st.selectbox("Select workspace:", workspaces_df['name'].tolist(), key="del_workspace")
        
        if st.button("🗑️ Delete Workspace", use_container_width=True):
            workspace_id = workspaces_df[workspaces_df['name'] == workspace_to_delete]['id'].values[0]
            conn = sqlite3.connect('users.db', check_same_thread=False)
            c = conn.cursor()
            c.execute("DELETE FROM workspaces WHERE id=?", (workspace_id,))
            c.execute("DELETE FROM feedback WHERE workspace_id=?", (workspace_id,))
            c.execute("DELETE FROM models WHERE workspace_id=?", (workspace_id,))
            conn.commit()
            conn.close()
            st.success(f"✅ Workspace '{workspace_to_delete}' deleted!")
            st.rerun()
    
    with col2:
        st.markdown("**Download Workspace Data:**")
        workspace_to_download = st.selectbox("Select workspace:", workspaces_df['name'].tolist(), key="dl_workspace")
        
        if st.button("📥 Download Data", use_container_width=True):
            workspace_id = workspaces_df[workspaces_df['name'] == workspace_to_download]['id'].values[0]
            conn = sqlite3.connect('users.db', check_same_thread=False)
            feedback_df = pd.read_sql_query(
                "SELECT * FROM feedback WHERE workspace_id=?",
                conn, params=(workspace_id,))
            conn.close()
            
            csv = feedback_df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"workspace_{workspace_to_download}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_dataset_management():
    """Dataset Management Section"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Dataset Management")
    
    # Get workspace selector
    conn = sqlite3.connect('users.db', check_same_thread=False)
    workspaces_df = pd.read_sql_query("SELECT id, name FROM workspaces", conn)
    
    if len(workspaces_df) == 0:
        st.info("No workspaces available")
        conn.close()
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    selected_workspace = st.selectbox("Select Workspace:", workspaces_df['name'].tolist(), key="dataset_workspace")
    workspace_id = workspaces_df[workspaces_df['name'] == selected_workspace]['id'].values[0]
    
    # Get dataset info
    feedback_df = pd.read_sql_query(
        "SELECT * FROM feedback WHERE workspace_id=?",
        conn, params=(workspace_id,))
    conn.close()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(feedback_df))
    with col2:
        st.metric("Positive", len(feedback_df[feedback_df['sentiment'] == 'Positive']))
    with col3:
        st.metric("Neutral", len(feedback_df[feedback_df['sentiment'] == 'Neutral']))
    with col4:
        st.metric("Negative", len(feedback_df[feedback_df['sentiment'] == 'Negative']))
    
    st.markdown("---")
    
    # Dataset preview
    st.markdown("### 📋 Dataset Preview")
    st.dataframe(feedback_df.head(10), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 🔧 Dataset Actions")
    
    col_act1, col_act2 = st.columns(2)
    
    with col_act1:
        st.markdown("**Replace Dataset:**")
        replace_file = st.file_uploader("Upload new CSV:", type=['csv'], key="replace_dataset")
        
        if st.button("🔄 Replace Dataset", use_container_width=True):
            if replace_file:
                try:
                    new_df = pd.read_csv(replace_file)
                    st.success(f"✅ Loaded {len(new_df)} samples")
                    st.info("⚠️ This will replace all existing feedback in this workspace!")
                    
                    if st.button("Confirm Replace", use_container_width=True):
                        # Delete old feedback
                        conn = sqlite3.connect('users.db', check_same_thread=False)
                        c = conn.cursor()
                        c.execute("DELETE FROM feedback WHERE workspace_id=?", (workspace_id,))
                        conn.commit()
                        conn.close()
                        
                        st.success("✅ Dataset replaced! Please re-analyze the feedback.")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col_act2:
        st.markdown("**Download Dataset:**")
        csv = feedback_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Dataset CSV",
            data=csv,
            file_name=f"dataset_{selected_workspace}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_model_management():
    """Model Management Section"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Model Management")
    
    # Get all models
    conn = sqlite3.connect('users.db', check_same_thread=False)
    models_df = pd.read_sql_query("""
        SELECT m.id, w.name as workspace, m.version, m.accuracy, m.f1_score, 
               m.created_at, u.username as trained_by
        FROM models m
        JOIN workspaces w ON m.workspace_id = w.id
        JOIN users u ON m.trained_by = u.id
        ORDER BY m.created_at DESC
    """, conn)
    conn.close()
    
    if len(models_df) == 0:
        st.info("No models trained yet")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    st.dataframe(models_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    
    # Performance visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=models_df['created_at'],
        y=models_df['accuracy'],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#00F0FF')
    ))
    fig.add_trace(go.Scatter(
        x=models_df['created_at'],
        y=models_df['f1_score'],
        mode='lines+markers',
        name='F1 Score',
        line=dict(color='#FF6EC7')
    ))
    
    fig.update_layout(
        title="Model Performance Over Time",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E6E6FA'),
        xaxis_title="Date",
        yaxis_title="Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔧 Model Actions")
    
    model_id_to_action = st.selectbox("Select model:", models_df['id'].tolist(), key="model_action")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        if st.button("🗑️ Delete Model", use_container_width=True):
            conn = sqlite3.connect('users.db', check_same_thread=False)
            c = conn.cursor()
            c.execute("DELETE FROM models WHERE id=?", (model_id_to_action,))
            conn.commit()
            conn.close()
            st.success("✅ Model deleted!")
            st.rerun()
    
    with col_m2:
        if st.button("📥 Download Model Info", use_container_width=True):
            model_info = models_df[models_df['id'] == model_id_to_action].to_dict('records')[0]
            json_data = json.dumps(model_info, indent=2, default=str)
            st.download_button(
                label="💾 Download JSON",
                data=json_data,
                file_name=f"model_{model_id_to_action}_info.json",
                mime="application/json",
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("### ⚙️ Train / Retrain Sentiment Model")
    conn = sqlite3.connect('users.db', check_same_thread=False)
    workspaces_df = pd.read_sql_query("SELECT id, name FROM workspaces", conn)
    conn.close()
    
    if len(workspaces_df) == 0:
        st.info("No workspaces available for training.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    selected_workspace = st.selectbox(
        "Workspace:",
        workspaces_df['name'].tolist(),
        key="train_model_workspace"
    )
    workspace_id = workspaces_df[workspaces_df['name'] == selected_workspace]['id'].values[0]
    
    if st.button("🚀 Train Sentiment Model", use_container_width=True):
        with st.spinner("Training sentiment model..."):
            result = train_sentiment_model(workspace_id, st.session_state.user_id)
        if result.get("trained"):
            st.success(f"✅ Trained v{result['version']} | Acc: {result['accuracy']:.2f} | F1: {result['f1']:.2f}")
        else:
            st.warning(result.get("reason", "Training failed."))
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_activity_logs():
    """Activity Logs Section"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📜 Activity Logs")
    
    # Get all activity logs
    conn = sqlite3.connect('users.db', check_same_thread=False)
    logs_df = pd.read_sql_query("""
        SELECT a.id, u.username, w.name as workspace, a.action_type, 
               a.description, a.created_at
        FROM activity_logs a
        JOIN users u ON a.user_id = u.id
        LEFT JOIN workspaces w ON a.workspace_id = w.id
        ORDER BY a.created_at DESC
        LIMIT 100
    """, conn)
    conn.close()
    
    if len(logs_df) == 0:
        st.info("No activity logs yet")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Filter options
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        action_filter = st.multiselect(
            "Filter by action:",
            logs_df['action_type'].unique().tolist(),
            default=logs_df['action_type'].unique().tolist()
        )
    with col_f2:
        user_filter = st.multiselect(
            "Filter by user:",
            logs_df['username'].unique().tolist(),
            default=logs_df['username'].unique().tolist()
        )
    
    # Apply filters
    filtered_logs = logs_df[
        (logs_df['action_type'].isin(action_filter)) &
        (logs_df['username'].isin(user_filter))
    ]
    
    st.dataframe(filtered_logs, use_container_width=True, hide_index=True)
    
    # Activity statistics
    st.markdown("---")
    st.markdown("### 📊 Activity Statistics")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        upload_count = len(logs_df[logs_df['action_type'] == 'upload'])
        st.metric("Uploads", upload_count)
    
    with col_s2:
        retrain_count = len(logs_df[logs_df['action_type'] == 'retrain'])
        st.metric("Retrains", retrain_count)
    
    with col_s3:
        correction_count = len(logs_df[logs_df['action_type'] == 'correction'])
        st.metric("Corrections", correction_count)
    
    with col_s4:
        unique_users = logs_df['username'].nunique()
        st.metric("Active Users", unique_users)
    
    st.markdown("</div>", unsafe_allow_html=True)

def log_activity(user_id, workspace_id, action_type, description):
    """Log user activity"""
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("""INSERT INTO activity_logs (user_id, workspace_id, action_type, description, created_at)
                 VALUES (?, ?, ?, ?, ?)""",
              (user_id, workspace_id, action_type, description, current_timestamp()))
    conn.commit()
    conn.close()

# Main App Flow
def main():
    load_css()
    init_db()
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        login_page()
    else:
        dashboard()
    
    st.markdown("""
    <div class='footer'>
        🎯 FeedbackAI Analyzer | AI-Powered Aspect-Based Sentiment Analysis | © 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()