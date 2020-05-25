#Proyect Iris

#Import libraries
import numpy as np
import pandas as pd

#4) Read data from iris.csv
# Read with the pandas method read_csv and assign to df_iris
df_iris = pd.read_csv("./DataSet/iris.csv")

# Print data head and shape
print(df_iris.head())
print(df_iris.shape)


print("\n4) Data Read")

# ¿Cuántos registros hay en los datos?
# Count by the length of the index from the data set
reg_num = len(df_iris.index)
print(" - There are " + str(reg_num) + " registers")

#¿Cuáles son las columnas presentes en los datos?
# show the columns names with the method .columns
row_names = df_iris.columns.values
#print(" - The columns in the data set are: " + str(row_names).replace("Index([","").replace("dtype='object')","").replace("],",""))
print(" - The columns in the data set are: " + str(row_names))

#¿Cuál es la columna objetivo, es decir, la que queremos predecir?
# use method index in the array columns and in the list row_names (cast) to get the index and the name of the header species
print(" - The target column is the " + str(row_names.tolist().index('species') + 1) + "th, " + str(df_iris.columns[row_names.tolist().index('species')]))

#¿Cuántos tipos de plantas hay?
# Counting using methods count and groupby
plants_types = len(df_iris.groupby(['species']).count())
plants = df_iris.groupby(['species']).count().index.values
print(" - There are " + str(plants_types) + " kinds of plants: " + str(plants))

#5) exploratory analysis
print("\n\n--------------------------\n5) Data Analysis")

#Métricas estadísticas básicas de cada columna, como la media o desviación típica.
# Describe data from the whole dataset and grouped by each specie in the 4 indicators 

df_statistics_sl = df_iris.groupby('species')['sepal_length'].describe()
df_statistics_sw = df_iris.groupby('species')['sepal_width'].describe()
df_statistics_pl = df_iris.groupby('species')['petal_length'].describe()
df_statistics_pw = df_iris.groupby('species')['petal_width'].describe()

print(" - General statistics:")
print(df_iris.describe())
print("\n Sepal length description:")
print(df_statistics_sl)
print("\n Sepal width description:")
print(df_statistics_sw)
print("\n Petal length description:")
print(df_statistics_pl)
print("\n Petal width description:")
print(df_statistics_pw)

#Gráficas para obtener un mejor conocimiento de los datos, por ejemplo: scatter plots entre pares de variables, distribuciones de cada variable
