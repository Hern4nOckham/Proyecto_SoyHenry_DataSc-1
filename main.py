# Librerias
from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
import json
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from collections import Counter


# Importamos los archivos
df_f1y2 = pd.read_parquet('3_Datasets\df_f1yf2.parquet')
df_f34y5 = pd.read_parquet('3_Datasets\df_f3f4yf5.parquet')
df_f6_ml = pd.read_parquet('3_Datasets\df_f6_ml.parquet')

# Funciones 1 y 2

# Convertimos a lista la columna "genre"
def convert_to_list(genres_array):
    return [element.strip("'") for element in genres_array]
# Aplicamos la funcion a la columna "genres"
df_f1y2['genres'] = df_f1y2['genres'].apply(convert_to_list)

# Lista de generos unicos
unique_genres = set()
for genres_list in df_f1y2['genres']:
    unique_genres.update(genres_list)
unique_genres_list = list(unique_genres)

# Funcion para normalizar.
def normalize_list_of_words(sentences_list):
   
    normalized_sentences = []
    for sentence in sentences_list:
        words = sentence.split()  
        normalized_words = [word.capitalize() for word in words]  
        normalized_sentence = ' '.join(normalized_words)  
        normalized_sentences.append(normalized_sentence)
    return normalized_sentences

# Funcion para normalizar.
def capitalize_first_words_in_sentence(sentence):
    
    words = sentence.split()  
    capitalized_words = [word[0].capitalize() + word.lower()[1:] for word in words]  
    return ' '.join(capitalized_words)  

# Crear una instancia FastAPI
app = FastAPI()

#This is our port
#http://127.0.0.1:8000

#Presentacion
@app.get("/")
def Presentation():
    return {"Hola, soy Hernán Pizarro. Les presento mi primer proyecto individual de Data Science"}

# Endpoint 1
@app.get("/PlayTimeGenre/{genre}")
def PlayTimeGenre(genre: str):

    # Seleccionamos las columnas que vamos a utilizar
    df_f1 = df_f1y2[['genres','release_year','sum_playtime_forever']]

    # Normalizamos
    normalized_genres = normalize_list_of_words(unique_genres_list)

    # Aseguramos de lo que se ingresen sea un estring
    if isinstance(genre, str):
        
        norm_genre = capitalize_first_words_in_sentence(genre)

        if norm_genre in normalized_genres:

            genre_to_find = norm_genre

            if norm_genre == 'Free To Play':
                genre_to_find = 'Free to Play'
            elif norm_genre == 'Rpg':
                genre_to_find = 'RPG'

            mask = df_f1['genres'].apply(lambda x: genre_to_find in x)

            df_f1_by_genre = df_f1[mask]

            grouped_df_f1_by_year = df_f1_by_genre.groupby('release_year')['sum_playtime_forever'].sum().reset_index()

            df1genre = grouped_df_f1_by_year[['release_year','sum_playtime_forever']]

            df1genre.sort_values(by='sum_playtime_forever', ascending=False, inplace=True)

            df1genre.reset_index(drop=True, inplace=True)

            year_most_hours_played = df1genre.iloc[0,0]
            max_sum_playtime_forever = df1genre.iloc[0,1]

            # Respuesta
            return {f"Año de lanzamiento con más horas reproducidas por género {genre_to_find}": int(year_most_hours_played)}
            
        else:
            return "El género ingresado no es válido. Inténtalo de nuevo."
    else:
        return "El género ingresado no es válido. Inténtalo de nuevo."


# Endpoint 2
@app.get("/UseForGenre/{genre}")
def UseForGenre(genre:str):


    # Seleccionamos las columnas que vamos a utilizar
    df_f2 = df_f1y2[['user_id','genres','release_year','sum_playtime_forever']]

    normalized_genres = normalize_list_of_words(unique_genres_list)

    if isinstance(genre, str):
    
        norm_genre = capitalize_first_words_in_sentence(genre)

        if norm_genre in normalized_genres:

            genre_to_find = norm_genre

            if norm_genre == 'Free To Play':
                genre_to_find = 'Free to Play'
            elif norm_genre == 'Rpg':
                genre_to_find = 'RPG'

            mask = df_f2['genres'].apply(lambda x: genre_to_find in x)

            df_f2_by_genre = df_f2[mask]

            grouped_df_f2_by_user_year = df_f2_by_genre.groupby(['user_id','release_year'])['sum_playtime_forever'].sum().reset_index()

            grouped_df_f2_by_user = df_f2_by_genre.groupby(['user_id'])['sum_playtime_forever'].sum().reset_index()

            grouped_df_f2_by_user.sort_values(by='sum_playtime_forever', ascending=False, inplace=True)

            grouped_df_f2_by_user.reset_index(drop=True, inplace=True)

            user_most_hours_played = grouped_df_f2_by_user.iloc[0,0]

            mask = grouped_df_f2_by_user_year['user_id'] == user_most_hours_played
            resultF2 = grouped_df_f2_by_user_year[mask]

            playtime_list = resultF2.rename(columns={'release_year': 'Release Year', 'sum_playtime_forever': 'Hours'})[['Release Year', 'Hours']].to_dict(orient='records')

            # Respuesta
            return {
                f"Usuario con más horas reproducidas por Género {genre_to_find}": user_most_hours_played,
                "Tiempo de juego": playtime_list
            }
        else:
            return "El género ingresado no es válido. Inténtalo de nuevo."
    else:
        return "El género ingresado no es válido. Inténtalo de nuevo."

# Endpoint 3
@app.get("/UsersRecommend/{year}")
def UsersRecommend(year:int):

    # Seleccionamos las columnas que vamos a utilizar
    df_f3 = df_f34y5[['item_id','title','recommend','sentiment_analysis','review_year']]

    mask = (df_f3['recommend'] == True) & (df_f3['sentiment_analysis'] != 0)
    df_f3f = df_f3[mask]
    df_f3 = df_f3f.reset_index(drop=True)

    if type(year) == int: 
        
        mask = df_f3['review_year'] == year
        df_f3_review_year = df_f3[mask].reset_index(drop=True)

        grouped_df_f3_review_year = df_f3_review_year.groupby(['title'])['sentiment_analysis'].sum().reset_index()

        grouped_df_f3_review_year.sort_values(by='sentiment_analysis', ascending=False, inplace=True)

        if not grouped_df_f3_review_year.empty:
            #Rank 1
            item_Rank_1 = grouped_df_f3_review_year.iloc[0,0]
            #Rank 2
            item_Rank_2 = grouped_df_f3_review_year.iloc[1,0]
            #Rank 3
            item_Rank_3 = grouped_df_f3_review_year.iloc[2,0]

            dataF3 = {
                'Rank': ['Position 1', 'Position 2', 'Position 3'],
                'title': [item_Rank_1, item_Rank_2, item_Rank_3]
            }

            # Respuesta
            return [{"Recomendacion " + str(index + 1): item} for index, item in enumerate(dataF3['title'])]

        else:
            return "El anio insertado no tiene resenas para calcular el ranking de los articulos mas recomendados. Intentelo con otro anio."
    else:
        return "Introduzca un anio valido como número entero."

# Endpoint 4
@app.get("/UsersNotRecommend/{year}")
def UsersNotRecommend(year:int):

    # Seleccionamos las columnas a utilizar
    df_f4 = df_f34y5[['item_id','title','recommend','sentiment_analysis','review_year']]

    mask = (df_f4['recommend'] == False) & (df_f4['sentiment_analysis'] == 0)
    df_f4f = df_f4[mask]
    df_f4 = df_f4f.reset_index(drop=True)

    if type(year) == int: 
        
        mask = df_f4['review_year'] == year
        df_f4_review_year = df_f4[mask].reset_index(drop=True)

        if not df_f4_review_year.empty:

            counter = df_f4_review_year['title'].value_counts()

            top_3_least_recommended = counter.head(3)

            #Respuesta
            return [{"Recomendacion " + str(index + 1): item} for index, item in enumerate(top_3_least_recommended.index)]
            
        else:
            return "El año insertado no tiene reseñas para calcular el ranking de los artículos más recomendados. Inténtelo con otro año."
    else:
        return "Introduzca un año válido como número entero."


# Endpoint 5
@app.get("/Sentiment_analysis/{year}")
def Sentiment_analysis(year:int):

    # Seleccionamos las columnas a utilizar
    df_f5 = df_f34y5[['sentiment_analysis','release_year']]

    if type(year) == int: 
        
        mask = df_f5['release_year'] == year
        df_f5_review_year = df_f5[mask].reset_index(drop=True)

        if not df_f5_review_year.empty:
        
            counter = df_f5_review_year['sentiment_analysis'].value_counts().sort_index()

            # Respuesta
            return {
                "Negativo": int(counter.get(0, 0)),
                "Neutral": int(counter.get(1, 0)),
                "Positivo": int(counter.get(2, 0))
            }

        else:
            return "El año insertado no tiene revisiones para calcular las categorías de análisis de sentimiento. Inténtelo con otro año."
    else:
        return "Por favor inserte un año válido como número entero."

# Endpoint 6
@app.get("/Game_Recommendation/{item_id}")
def Game_Recommendation(item_id:int):
   
    # Combinacion columnas de texto en una sola columna para vectorización
    df_f6_ml['combined_features'] = df_f6_ml['tags']

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_f6_ml['combined_features'].fillna(''))

    num_dimensions = tfidf_matrix.shape[1]

    num_hash_functions = 4 

    engine = Engine(num_dimensions, lshashes=[RandomBinaryProjections('rbp', num_hash_functions)])

    for i, row in enumerate(tfidf_matrix):

        game_id = df_f6_ml.iloc[i]['item_id']

        engine.store_vector(row.toarray().flatten(), data=game_id)

    if (type(item_id) == float) | (type(item_id) == int):

        if item_id in df_f6_ml['item_id'].values:
            
                query = tfidf_matrix[df_f6_ml['item_id'] == item_id].toarray().flatten()

                neighbors = engine.neighbours(query)

                recommended_game_ids = [neighbor[1] for neighbor in neighbors if neighbor[1] != item_id][:5]

                recommended_games = df_f6_ml[df_f6_ml['item_id'].isin(recommended_game_ids)][['title']]

                result = [{'Recomendacion {}'.format(i + 1): game} for i, game in enumerate(recommended_games['title'])]

                return result
        else:
            return "El item_id no existe. Inténtalo de nuevo."
    else:
        return "El item_id debe ser un número; ingrese un nuevo item_id"