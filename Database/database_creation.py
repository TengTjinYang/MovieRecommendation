import pandas as pd
import re
import tensorflow_hub as hub
import numpy as np
import sqlite3



# Load the Universal Sentence Encoder model from TensorFlow Hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
use_model = hub.load(module_url)


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
    return pd.merge(ds2, ds1, on= column, how='left')

#define a custom tokenizer
def combined_text_tokenizer(text):
    # Replace spaces with underscores
    text = re.sub(r'\s', '_', text)
    return [text]

#getting USE embeddings for a list of sentences
def get_sentence_embeddings(sentences):
    embeddings = use_model(sentences)
    return embeddings.numpy()

def vectorise_cast_crew(dataset,category):
    # Fill NaN values with an empty string and concatenate 'primaryName' and 'characters' columns
    dataset['actor_crew'] = dataset['primaryName'].fillna('') + ' as ' + dataset[category].fillna('')

    # Group by 'tconst' and join the text data for each 'tconst'
    film_chars = dataset.groupby('tconst')['actor_crew'].apply(lambda x: ' '.join(x)).reset_index()

    # Create TF-IDF vectorizer
    #vectorizer = TfidfVectorizer(tokenizer=combined_text_tokenizer)

    # Apply TF-IDF vectorization on the combined text for each 'tconst'
    #tfidf_vectors = vectorizer.fit_transform(film_chars['actor_char'])

    # Create a DataFrame to store 'tconst' with its respective TF-IDF vectors
    #tconst_vector_df = pd.DataFrame(tfidf_vectors.toarray(), columns=vectorizer.get_feature_names_out(), dtype='float64')
    #tconst_vector_df.insert(0, 'tconst', film_chars['tconst'])  # Add 'tconst' column to the vector DataFrame
    
    
    film_chars = dataset.groupby('tconst')['actor_crew'].apply(lambda x: ' '.join(x)).reset_index()

    # Get the USE embeddings for the concatenated text
    sentence_embeddings = get_sentence_embeddings(film_chars['actor_crew'])

    # Create a DataFrame to store 'tconst' with its respective USE embeddings
    embedding_columns = [f"use_{i}" for i in range(len(sentence_embeddings[0]))]  # Column names for embeddings
    vectors = pd.DataFrame(np.vstack(sentence_embeddings), columns=embedding_columns)
    vectors.insert(0, 'tconst', film_chars['tconst'])  # Add 'tconst' column to the embedding DataFrame
    return vectors

# Now, 'tconst_vector_df' contains TF-IDF vectors for each 'tconst' while retaining actor and character information
# You can merge or join this DataFrame with other datasets using 'tconst' as a common key

# merge it into one dataset using the identifier
# take it apart for each row and vetorise the columns
# merge the vetorised sections into one vector
# store it in the database


####  ---------------------------------   MAIN BODY    ---------------------------------#### 

#loading all the csv files
title_basics = load_csv("test/ImdbTitleBasicsTest.csv")
name = load_csv("test/ImdbNameTest.csv")
title_ratings = load_csv("test/ImdbTitleRatingsTest.csv")
title_principals = load_csv("test/ImdbTitlePrincipalsTest.csv")

# if you want to display the full dataset or not 
pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

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
title_priciples_actors, title_principals_crew = split_dataset_by_value(title_principals,'category','actor','self')

#cleaning up the two split tables
title_principals_crew = remove_columns(title_principals_crew,'characters')
title_priciples_actors = clean_dataset(title_priciples_actors,'characters', r'\\n|\n', 'unknown')

# removing the [""] from the title_pricipals dataset
title_priciples_actors['characters'] = title_priciples_actors['characters'].str.strip('[]').str.replace('"', '')
title_priciples_actors = standardise_dataset_principals(title_priciples_actors,'category','characters','self')

#merging basics and ratings
merged_dataset_films = merge_2_datasets(title_basics,title_ratings,'tconst')

##vectorise basics and ratings merged dataset

print(title_ratings)
print(title_basics)
print(vectorise_cast_crew(title_priciples_actors,'characters'))
print(vectorise_cast_crew(title_principals_crew,'category'))


