#Proyect Iris

#Import libraries
import numpy as np
import pandas as pd

#4) Read data from iris.csv
# Read with the pandas method read_csv and assign to pd_iris
pd_iris = pd.read_csv("./DataSet/iris.csv")
print("4) ")

# ¿Cuántos registros hay en los datos?
# Count by the length of the index from the data set
reg_num = len(pd_iris.index)
print(" - There are " + str(reg_num) + " registers")

#¿Cuáles son las columnas presentes en los datos?
# show the columns names with the method .columns
row_names = pd_iris.columns.values
#print(" - The columns in the data set are: " + str(row_names).replace("Index([","").replace("dtype='object')","").replace("],",""))
print(" - The columns in the data set are: " + str(row_names))

#¿Cuál es la columna objetivo, es decir, la que queremos predecir?
print(" - The target column is the " + str(row_names.tolist().index('species') + 1) + "th, " + str(pd_iris.columns[4]))

#¿Cuántos tipos de plantas hay?
plants_types = len(pd_iris.groupby(['species']).count())
plants = pd_iris.groupby(['species']).count().index.values
print(" - There are: " + str(plants_types) + " kinds of plants: " + str(plants))