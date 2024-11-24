import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tkinter.constants import X

# Se obtiene una 
def get_entity_contexts_with_offsets(df, text_dict, window):
    contexts = []
    for index, row in df.iterrows():
        text_id = row['article_id']
        start_offset = int(row['start_offset'])
        end_offset = int(row['end_offset'])
        entity_text = row['entity_mention']
        main_role = row['main_role']
        fine_grained_roles = row['fine-grained_roles']

        text = text_dict.get(text_id)

        if text:
            # Convert text to string if it's a NumPy array
            if isinstance(text, np.ndarray):  # Check if text is a NumPy array
                text = text.astype(str)
                text = " ".join(text)

            # Extract context using the sliding window method from the given offsets
            words = text.split()
            entity_start = len(text[:start_offset].split())
            entity_end = len(text[:end_offset].split())

            context_start = max(0, entity_start - window)
            context_end = min(len(words), entity_end + window)
            context = " ".join(words[context_start:context_end])
            contexts.append({
                'text_id': text_id,
                'entity': entity_text,
                'start_offset': start_offset,
                'end_offset': end_offset,
                'context': context,
                'main_role': main_role,
                'fine_grained_roles': fine_grained_roles

            })
    return pd.DataFrame(contexts)


# Aplicando CounterVectorizer
def get_vectorizer(embedding_method):
  # 3. Vectorización del texto
  if embedding_method == "countvectorizer":
      vectorizer = CountVectorizer()
  elif embedding_method == "tfidf":
      from sklearn.feature_extraction.text import TfidfVectorizer
      vectorizer = TfidfVectorizer()
  else:
      raise ValueError("Método no válido. Usa 'countvectorizer' o 'tfidf'.")

  return vectorizer


# Vectorizar cada texto de contexto
def vectorizar_cada_contexto(df_context, vectorizer):
    x_train = []
    y_train = []
    
    for index, row in df_context.iterrows():

        context = row['context']

        vector = vectorizer.transform([context])

        X_train = vector.toarray()
        x_train.append(X_train[0])

        y_train.append(row['main_role'])

    return x_train, y_train