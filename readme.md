#  **<p align="center">Proyecto Individual N° 1 de Data Science</p>** 

<img src="5_Imagenes\juegossteam.jpg" width="1010" height="300">

# <p align="center">Machine Learning Operation - MLOps</p>


## *1. Descripción del Proyecto*

Proyecto individual N° 1 de la etapa de labs de la carrera de Data Science de la academia SoyHenry.

El **objetivo de este proyecto** es desarrollar un proceso de Data Engineering y aplicar técnicas de Machine Learning para analizar un dataset de la plataforma digital de videojuegos Steam que posee 75 millones de cuentas de ususarios activas.

## *2. Requerimientos del Proyecto*

Se debe crear un sistema de recomendación en base a las siguientes funciones:

- **PlayTimeGenre**

- **UserForGenre**

- **UsersRecommend**

- **UsersNotRecommend**

- **Sentiment_analysis**
 
- **Recomendacion_juego**

[Requerimientos del proyecto](https://github.com/soyHenry/PI_ML_OPS/tree/PT)

## *3. Desarrollo del Proyecto*

El proyecto se llevó a cabo en tres etapas:

1° Etapa: se realizó la extración y transformación del datasets que brindo Henry.

2° Etapa: se realizarón los análisis y gráficos de los datos.

3° Etapa: se llevó a cabo la visualización de los datos y el sistema de recomendación.

Notebooks:

**1_xETL**

A los archivos originales steam_games.json.gz, user_reviews.json.gz y user_items.json.gz, se los abrio, descomprimio, EDA (Analisis Exploratorio de los Datos), ETL (Extraccion Transformación y Carga) y NLP (Procesamiento del Lenguaje Natural). Nos permitio explorar el conjunto de datos y crear archivos .csv para ser procesados posteriormente. 

**2.1_xFunciones**

Usando los archivos .csv creados anteriormente se realizó un ETL para generar un archivo .parquet donde tomaran la información las funciónes PlayTimeGenre y UserForGenre.

**2.2_xFunciones**

Usando los archivos .csv  en el "1 EDA_ETL_Desanidar_NLP.ipynb" se realizó un ETL para generar un archivo .parquet donde tomaran la información las funciónes UsersRecommend, UsersNotRecommend y Sentiment_analysis.

**2.3_xFunciones**

Usando los archivos .csv  en el "1 EDA_ETL_Desanidar_NLP.ipynb" se realizó un ETL para generar un archivo .parquet donde toma información la función Recomendacion_juego.

**3_xEDA**

Usando los archivos .parquet se realizó un EDA resultando gráficos de barras, dispersión y nube de palabras.

**4_xFunciones_ML**

Se desarrollaron las funciones PlayTimeGenre, UserForGenre, UsersRecommend, UsersNotRecommend, Sentiment_analysis y Recomendation_Game (ML).


## *4. Datasets*

Fueron brindados por Henry tres archivos en formato .json y comprimido .gz:
[Datasets](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj)

## *5. Consideraciones finales*

Durante el proceso del trabajo se afinzaron muchos conociminetos en cuanto a la limpieza de datos, su transformación y el armado de un modelo de Machine Learning. Por mi parte, fue un excelente proceso de aprendizaje y una oportunidad de aplicar las herramientas aprendidas durante los modúlos teniendo en cuenta que partia de una base sin conocimientos en programación y en ciencia de datos.

En cuanto al **análisis de los datos** se puede decir: 

- Que los juegos que mas horas se jugaron fueron los lanzados en el año 2012, seguido de los lanzados en el año 2013 y 2015.

- Mientras mas reseñas positiva tiene un juego, más usuarios juegan con mayor cantidad de tiempo.

- Las palabras que mas se repiten en los títulos de los videos juegos son:

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
