import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sentence_transformers import SentenceTransformer



#title pricipals combine category and character columns
###Basic Plan of Action###
# loading one csv file
def load_csv(file_name):
    return pd.read_csv(file_name)

#remove tvSeries or tvEpisode from the datasets - only want data about movies/ shorts
def filter_dataset_basics(dataset, type):
    condition = dataset['titleType'].str.strip() != type
    return dataset[condition]


# Get rid of columns that are not needed
def remove_columns(dataset, column_name):
    return dataset.drop( column_name, axis=1)


# Lowercase all values in the dataset
def dataset_lowercase(dataset):
    return dataset.apply(lambda x: x.astype(str).str.lower() if x.dtype == 'object' else x)

# process the csv file and fill in missing data with default values: try to do this in one method for all files.
def clean_dataset(dataset, column, value, replacement):
    dataset[column] = dataset[column].replace(value, replacement, regex=True)
    return dataset



#looking for problematic rows in the dataset based on no value in the genre column
def identify_rows_with_issue(dataset, column):
    mask = dataset[column].isnull() | (dataset[column] == 'nan')
    
    # Use boolean indexing to filter and print the rows with missing values
    rows_with_missing_values = dataset[mask]
    
    print(f"Rows with missing values in the '{column}' column:")
    print(rows_with_missing_values)

#splitting the dataset - for title principles to split between actors and crew
def split_dataset_by_value(dataset, column, value1,value2):
    # Create a boolean mask for rows with the specific value in the specified column
    mask = dataset[column].isin([value1, value2])
    
    # Create two dataframes based on the boolean mask
    value_df = dataset[mask]
    value_not_df = dataset[~mask]  # ~ = negation operator
    
    return value_df, value_not_df

#replacing the charcter played with self if it is empty and the category is self
def standardise_dataset_principals(dataset,column1,column2, value):
    mask = (dataset[column1] == value) & (dataset[column2] == 'unknown')
    dataset.loc[mask, 'characters'] = 'self'
    return dataset

#merge the ratings and basics dataset
def merge_2_datasets(ds1,ds2,column):
    return ds1.merge(ds2, on= column, how='left')

# merge it into one dataset using the identifier
# take it apart for each row and vetorise the columns
# merge the vetorised sections into one vector
# store it in the database


####  ---------------------------------   MAIN BODY    ---------------------------------#### 

#create a database to store the final vectors
# Connect to SQLite database or create a new one if it doesn't exist
# conn = sqlite3.connect('vectors_database.db')

# # Create a cursor object to execute SQL queries
# cursor = conn.cursor()

# # Create a table to store vectors
# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS Vectors (
#         tconst TEXT PRIMARY KEY,
#         vector_dim1 REAL,
#         vector_dim2 REAL,
#         vector_dim3 REAL,
#         -- Add columns for each dimension of the vector
#         -- Add as many columns as the dimensions of your vectors require
#         -- (For example, if your vectors are 100-dimensional, add 100 columns)
#     )
# ''')

# #need to add all the vectors from the panda dataframe

# # Save changes and close the connection
# conn.commit()
# conn.close()

#loading all the csv files
title_basics = load_csv("test/ImdbTitleBasicsTest.csv")
name = load_csv("test/ImdbNameTest.csv")
title_ratings = load_csv("test/ImdbTitleRatingsTest.csv")
title_principals = load_csv("test/ImdbTitlePrincipalsTest.csv")

# if you want to display the full dataset or not 
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

#removing unwanted columns
name = remove_columns(name,'birthYear')
name = remove_columns(name,'deathYear')
name = remove_columns(name,'primaryProfession')


title_principals = remove_columns(title_principals,'job')
title_principals = remove_columns(title_principals,'ordering')



title_basics = remove_columns(title_basics,'originalTitle')
title_basics = remove_columns(title_basics,'endYear')

#setting the datasets to lowercase
title_basics = dataset_lowercase(title_basics)
title_ratings = dataset_lowercase(title_ratings)
title_principals = dataset_lowercase(title_principals)
name = dataset_lowercase(name)

# removing the rows with tvseries ans tvepisode from the title_basics dataframe
title_basics = filter_dataset_basics(title_basics,'tvseries')
title_basics = filter_dataset_basics(title_basics,'tvepisode')

#filling in missing information for title_basics
title_basics = clean_dataset(title_basics,'runtimeMinutes', r'\\n|\n', 0)
title_basics = clean_dataset(title_basics,'genres','nan', 'unknown')
title_basics = clean_dataset(title_basics,'genres', r'\\n|\n', 'unknown')

#filling in missing information or wrong information for title_principals
title_principals = clean_dataset(title_principals,'category','actress', 'actor')


#splitting up title_principles into actors and crew
merged_dataset_people = merge_2_datasets(title_principals,name,'nconst')
merged_dataset_people = remove_columns(merged_dataset_people,'knownForTitles')

merged_dataset_people = clean_dataset(merged_dataset_people,'characters', r'\\n|\n', 'unknown')

# removing the [""] from the title_pricipals dataset
merged_dataset_people['characters'] = merged_dataset_people['characters'].str.strip('[]').str.replace('"', '')
merged_dataset_people = standardise_dataset_principals(merged_dataset_people,'category','characters','self')

print(merged_dataset_people)
##vectorise basics and ratings merged dataset

merged_dataset_people['primaryName'].fillna('', inplace=True)

merged_dataset_people['text_to_vectorize'] = merged_dataset_people.apply(
    lambda row: row['characters'] + ' ' + row['primaryName'] 
                if row['category'] in ['actor', 'self'] 
                else row['primaryName'], axis=1)

# Vectorize the 'text_to_vectorize' column
model = SentenceTransformer('all-MiniLM-L6-v2')
text_vectors = model.encode(merged_dataset_people['text_to_vectorize'].tolist())

# Vectorize 'category' using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
category_encoded = encoder.fit_transform(merged_dataset_people[['category']])

# Concatenate all vectors (text and categorical)
final_vectors = [np.concatenate((tv, ce)) for tv, ce in zip(text_vectors, category_encoded)]

# Convert the list of vectors into a DataFrame
vectors_df = pd.DataFrame(final_vectors)

# Add the 'tconst' column from merged_dataset to vectors_df
vectors_df['tconst'] = merged_dataset_people['tconst'].values

# Rearrange the columns so 'tconst' is the first column
columns = ['tconst'] + [col for col in vectors_df.columns if col != 'tconst']
vectors_df = vectors_df[columns]

print(vectors_df)

# print(title_ratings)
# print(title_basics)
# print(vectorise_cast_crew(title_priciples_actors,'characters'))
# print(vectorise_cast_crew(title_principals_crew,'category'))


