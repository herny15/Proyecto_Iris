#Proyect Iris

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns




#Function case to select a variable from the dataset
def data_var(i):
        switcher={
                1:'petal_length',
                2:'petal_width',
                3:'sepal_length',
                4:'sepal_width',
                5:'setosa',
                6:'versicolor',
                7:'virginica',
                8:'All',
             }
        return switcher.get(i,"Invalid variabel (1,2,3,4 - 5,6,7)")

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
# Scatter plots
print("Enter the first variable to compare\n1 - Petal Length\n2 - Petal Width\n3 - Sepal Length\n4 - Sepal Width")
x_var = data_var(int(input()))

print("Enter the second variable to compare\n1 - Petal Length\n2 - Petal Width\n3 - Sepal Length\n4 - Sepal Width")
y_var = data_var(int(input()))

print("Enter the kind of plant\n5 - Setosa\n6 - Versicolor\n7 - Virginica\n8 - All")
kind_var = data_var(int(input()))


if(kind_var == 'All'):
    df_iris.plot.scatter(x=x_var, y=y_var)
else:
    df_iris[df_iris.species == kind_var].plot.scatter(x=x_var, y=y_var)


plt.title(kind_var)
plt.show()

# Same plot with seaborn for all
sns.lmplot(x=x_var, y= y_var, data=df_iris, palette='Set1')
plt.show()

# Pie chart of species

df_iris.groupby('species')['species'].count().plot(kind='pie')
plt.title("Species count")
plt.show()

# Histogram
# num_bins = 15
# n1 = plt.hist(df_iris.sepal_length, num_bins)
# n2 = plt.hist(df_iris.sepal_width, num_bins)
# n3 = plt.hist(df_iris.petal_length, num_bins)
# n4 = plt.hist(df_iris.petal_width, num_bins)

for species_class in df_iris.groupby(['species']).count().index.values:
    df_iris[df_iris.species == species_class].plot(kind='kde')
    plt.title(species_class)
    
plt.show()