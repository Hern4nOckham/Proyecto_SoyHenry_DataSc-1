#  **<p align="center">Proyecto Individual N° 1 de Data Science</p>** 

<img src="Imagenes\juegossteam.jpg" width="1010" height="300">

# <p align="center">Machine Learning Operation - MLOps</p>


## *1. Descripción del Proyecto*

Proyecto individual N° 1 de la etapa de labs de la carrera de Data Science de la academia SoyHenry.

El objetivo de este proyecto es desarrollar un proceso de Data Engineering y aplicar técnicas de Machine Learning para analizar un dataset de la plataforma digital de videojuegos Steam que posee mas de 3.000 juegos y 75 millones de cuentas de ususarios activas.

Se implementará una API que ofrecerá endpoints para acceder a los resultados del análisis. 💪🎮

## *2. Requerimientos del Proyecto*

Se debe crear las siguientes funciones para los endpoints que se consumirán en la API.

- def **PlayTimeGenre**( genero : str ): Debe devolver año con mas horas jugadas para dicho género.

Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}

- def **UserForGenre**( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}

- def **UsersRecommend**( año : int ): Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)

Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

- def **UsersNotRecommend**( año : int ): Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)

Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

- def **Sentiment_analysis**( año : int ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.

Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}
 
Armar un sistema de recomendación. Para ello, te ofrecen dos propuestas de trabajo: En la primera, el modelo deberá tener una relación ítem-ítem, esto es se toma un item, en base a que tan similar esa ese ítem al resto, se recomiendan similares. Aquí el input es un juego y el output es una lista de juegos recomendados, para ello recomendamos aplicar la similitud del coseno. La otra propuesta para el sistema de recomendación debe aplicar el filtro user-item, esto es tomar un usuario, se encuentran usuarios similares y se recomiendan ítems que a esos usuarios similares les gustaron. En este caso el input es un usuario y el output es una lista de juegos que se le recomienda a ese usuario, en general se explican como “A usuarios que son similares a tí también les gustó…”. Deben crear al menos uno de los dos sistemas de recomendación

Si es un sistema de recomendación item-item:

 - def **Recomendacion_juego**( id de producto ): Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

Si es un sistema de recomendación user-item:

- def **Recomendacion_usuario**( id de usuario ): Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.

Crear un video menor a 7 minutos mostrando que las herramientas funcionan realmente. Para grabarlo, pueden usar la herramienta Zoom, haciendo una videollamada y grabando la pantalla.

Se considera un producto minimo viable aprobatorio si cumple con la siguiente grafica:

<img src="Imagenes\MNV.png" width="1010" height="300">

## *3. Desarrollo del Proyecto*

El proyecto cuanta con las carpetas:

- Datasets
- Notebooks
- Imagenes

Y archivos: 

- main.py
- README.md

Breve descripción de lo desarrollados en los Notebooks:

**1 EDA_ETL_Desanidar_NLP**

A los archivos originales steam_games.json.gz, user_reviews.json.gz y user_items.json.gz, se los abrio, descomprimio, EDA (Analisis Exploratorio de los Datos), ETL (Extraccion Transformación y Carga) y NLP (Procesamiento del Lenguaje Natural). Nos permitio explorar el conjunto de datos y crear archivos .csv para ser procesados posteriormente. 

**2.1 ETL_Funciones**

Usando los archivos .csv creados anteriormente se realizó un ETL para generar un archivo .parquet donde tomaran la información las funciónes PlayTimeGenre y UserForGenre.

**2.2 ETL_Funciones**

Usando los archivos .csv  en el "1 EDA_ETL_Desanidar_NLP.ipynb" se realizó un ETL para generar un archivo .parquet donde tomaran la información las funciónes UsersRecommend, UsersNotRecommend y Sentiment_analysis.

**2.3 ETL_Funciones**

Usando los archivos .csv  en el "1 EDA_ETL_Desanidar_NLP.ipynb" se realizó un ETL para generar un archivo .parquet donde toma información la función Recomendacion_juego.

**3 EDA_Graficos**

Usando los archivos .parquet se realizó un EDA resultando gráficos de barras, dispersión y nube de palabras.

<img src="Imagenes\graficobarras.png" width="1010" height="400">
Gráfico 1: Horas jugadas por año.
<br>
</br>
<img src="Imagenes\graficodispersion.png" width="1010" height="400">
Gráfico 2: Horas jugadas según las reseñas (negativas-neutras-positivas).
<br>
</br>
<img src="Imagenes\graficonubedepalabras.png" width="1010" height="800">
Gráfico 3: Palabras que mas se repiten en los títulos de los videos juegos.


<br>
</br>

**4 Funciones_ML**

Se desarrollaron las funciones PlayTimeGenre, UserForGenre, UsersRecommend, UsersNotRecommend, Sentiment_analysis y Recomendacion_juego (ML).

Este ultimó notebook se encuentra fuera de la carpeta "Notebooks" debido a que el trabajo en su mayor parte se realizó en Google Colab y al pasar a Visual Estudio Code, la unica forma de cargar el datasets era este archivo fuera de la carpeta. 


## *4. Datasets*

Fueron brindados por Henry tres archivos en formato .json y comprimido .gz:
- steam_games
- user_reviews
- user_items

En base a los archivos originales se crearon tres archivos en formato .parquet:
- df_f1yf2 : toma información las funciónes PlayTimeGenre y UserForGenre.
- df_f3f4yf5 : toma información las funciónes UsersRecommend, UsersNotRecommend y Sentiment_analysis.
- df_f6: toma información la función Recomendacion_juego (ML).


## *5. Conclusíones*

Durante el proceso del trabajo se afinzaron muchos conociminetos en cuanto a la limpieza de datos, su transformación y el armado de un modelo de Machine Learning. Por mi parte, fue un excelente proceso de aprendizaje y una oportunidad de aplicar las herramientas aprendidas durante los modúlos teniendo en cuenta que partia de una base sin conocimientos en programación y en ciencia de datos.

En cuanto al **análisis de los datos** se puede decir: 

- Que los juegos que mas horas se jugaron fueron los lanzados en el año 2012, seguido de los lanzados en el año 2013 y 2015 (Gráfico 1).

- Mientras mas reseñas positiva tiene un juego, más usuarios juegan con mayor cantidad de tiempo (Gráfico 3).

- Las palabras que mas se repiten en los títulos de los videos juegos son (Gráfico 3):

   - Soundtrack
   - Pack
   - Fantasy
   - Ground


**Deploy**
Para el deploy de la API, se utilizó la plataforma Render. Los datos están listos para ser consumidos y consultados a partir del siguiente link. [https://hernanpizarro.onrender.com/docs](https://hernanpizarro.onrender.com/docs)

**Video**
Para consultar sobre los pasos del proceso acceder al siguiente link. [https://youtu.be/8UopATbm--U](https://youtu.be/8UopATbm--U)


## *6. Autor*

- **Hernán Pizarro**: hern4npizarro@gmail.com | [Linkedin](https://www.linkedin.com/in/hern%C3%A1n-pizarro-683679268/)
