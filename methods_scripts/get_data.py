"""
    En este script se estará manejando las funciones para preprocesar los datos.
    Esto incluye:
        0. Creación de las estructuras de los datos
        1. Limpieza
        2. Estandarización
        3. Transformación a vectores
"""

# libraries
import numpy as np
import pandas as pd
import os

# Esta funcino es para crear arreglo de archivos
def get_files_array(dir = "training_data/EN/raw-documents/"):
    
    dir_en = dir
    all_files = os.listdir(dir_en)

    files_array = []
    names_files = []

    for file_name in all_files:
        try:
            file = open(dir_en + file_name, "r")
            content = file.read()
            names_files = np.append(names_files, file_name)
            files_array = np.append(files_array, content)
            file.close()
        except:
            continue

    files_array.shape

    # Reshape as a column vector
    files_array = files_array.reshape(files_array.shape[0], 1)
    return files_array, names_files



# Funcion para obtener el texto completo el cual sera vectorizado
def get_texto_completo():
    files_array, names_files = get_files_array()

    texto_completo = []
    articles_map = {}
    for i in range(len(names_files)):
        articles_map[names_files[i]] = files_array[i]
        texto_completo.append(str(files_array[i][0]))
    
    return texto_completo, articles_map


# Crea el dataframe
def get_df_entities(dir = "training_data/EN/subtask-1-annotations.txt"):
    annotations_1 = dir

    # Initialize a empty dataframe
    df_entities = pd.DataFrame(columns=['article_id','entity_mention','start_offset','end_offset','main_role','fine-grained_roles'])
    columns=['article_id','entity_mention','start_offset','end_offset','main_role','fine-grained_roles']

    with open(annotations_1, encoding='utf-8') as file:
        lines_text_1 = file.readlines()  # Read line by line

    for line in lines_text_1:
        line = line.strip()
        line = line.split("\t")
        new_row =  {columns[0]: line[0],
                    columns[1]: line[1],
                    columns[2]: line[2],
                    columns[3]: line[3],
                    columns[4]: line[4],
                    columns[5]: line[5:]}
        df_entities = pd.concat([df_entities, pd.DataFrame([new_row])], ignore_index=True)

    print("\nSize:",df_entities.shape)
    return df_entities


