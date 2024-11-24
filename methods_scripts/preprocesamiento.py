import spacy
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import string
import re

# Descargar las stopwords de NLTK si no lo has hecho aún
nltk.download("stopwords")

# Configuración inicial
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

# Diccionario de contracciones comunes
contractions = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", 
    "could've": "could have", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
    "i'd": "I would", "i'll": "I will", "i'm": "I am", "isn't": "is not", "it's": "it is",
    "let's": "let us", "ma'am": "madam", "mightn't": "might not", "mustn't": "must not",
    "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that's": "that is",
    "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "we'd": "we would", "we're": "we are", "weren't": "were not",
    "what's": "what is", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
    "you're": "you are", "you've": "you have"
}

# Función para expandir contracciones
def expand_contractions(text):
    words = text.split()
    expanded_words = [contractions[word] if word in contractions else word for word in words]
    return " ".join(expanded_words)

# Este preprocesamiento simple se hará en todo el texto completo
def preprocess_simple(text):
    # 1. Expandir contracciones
    text = expand_contractions(text)

    # 2. Convertir a lowercase
    text = text.lower()
    
    # 3. Eliminar números, caracteres especiales y puntuaciones
    text = re.sub(r'\d+', '', text)  # Eliminar números
    text = text.translate(str.maketrans("", "", string.punctuation))  # Eliminar puntuaciones
    text = re.sub(r'\W+', ' ', text)  # Eliminar caracteres especiales
    
    return text

# Este preprocesamiento se hace sobre cada contexto extraído de las entidades
def preprocess_context_text(text):
    # 1. Expandir contracciones
    text = expand_contractions(text)

    # 2. Convertir a lowercase
    text = text.lower()

    # 3. Tokenización con SpaCy y eliminación de números, puntuaciones, caracteres especiales y stop words
    doc = nlp(text)
    filtered_tokens = [
        token.text for token in doc 
        if token.text not in stop_words 
        and not token.is_punct 
        and not token.is_space 
        and not token.like_num  # Eliminar números
        and not re.search(r'\W', token.text)  # Eliminar caracteres especiales
    ]
    preprocessed_text = " ".join(filtered_tokens)

    return preprocessed_text
