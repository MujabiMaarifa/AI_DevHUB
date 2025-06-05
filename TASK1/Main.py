import pandas as pd
import matplotlib as mtlp 
import seaborn as sb 
import numpy as np 

# read the  contents of the csv file
data = pd.read_csv('iris/iris.data')
print(data)

#printing the first five values and columns and rows
print("\n... The first five rows in the dataset iris... \n")
filtered_data = data.head()
print(filtered_data)

#finding null values present in the data set

print("\n... The number of null values in filtered data set info is as follows ...\n")

null_values = filtered_data.info()
print(null_values)

# summary statistics

print("\n... This is the summary for the filtered set ...\n")

summary = filtered_data.describe()
print(summary)

# data visualization