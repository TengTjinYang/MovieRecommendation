from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
import sqlite3
import faiss
from sklearn.decomposition import PCA
import evaluate
import pandas as pd

#load csv file
def load_csv(file_name):
    return pd.read_csv(file_name)

#tokenise text
def tokenize(text):
    return text.lower().split()

# convert the llm output into vector
def vectorize_text(text):
    valid_words = [word for word in text if word in word2vec_model.key_to_index]
    if len(valid_words) > 0:
        return np.mean(word2vec_model[valid_words], axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)
    
# checking the dimentions of the prediction vector and adjusting accordingly
def adjust_vector_dimensions(prediction_vector, target_dimension):
    current_dimension = len(prediction_vector)

    if current_dimension < target_dimension:
        # Pad the vector with zeros to match the target dimension
        padded_vector = np.pad(prediction_vector, (0, target_dimension - current_dimension), 'constant')
        return padded_vector

    elif current_dimension > target_dimension:
        # Reduce dimensions using PCA to match the target dimension
        pca = PCA(n_components=target_dimension)
        reduced_vector = pca.fit_transform(prediction_vector.reshape(1, -1))
        return reduced_vector.flatten()

    else:
        # Dimensions already match, return the vector as is
        return prediction_vector

 #calculating the cosine similarity
def compute_cosine_similarity(vector1,vector2):
    # Compute dot product between prediction_vector and retrieved_vector
    dot_product = np.dot(vector1, vector2)

    # Compute norms of the vectors
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    # Compute and return cosine similarity
    return dot_product / (norm_vector1 * norm_vector2)

#computing the bert score
def bert_score(predictions,truth):
    bertscore = evaluate.load("bertscore")
    bert_score = bertscore.compute(predictions=predictions, references=truth, lang="en")
    print(f"BERTScore: {bert_score}")

# lsh the original query to our dataset - it should return a movie name
# convert the lsh output to vector
# compare the two vectors (lsh output and llm output) using cosine similarity


# -----------------------------------------  main body  ----------------------------------------#





def load_database_vectors(database_name):
    # Connect to the database
    conn = sqlite3.connect('movie_data.db')
    cursor = conn.cursor()

    # Execute a query to fetch all vectors
    cursor.execute("SELECT * FROM vectors")

    # Fetch all rows from the query result
    rows = cursor.fetchall()

    # Separate identifiers and vectors
    identifiers = [row[0] for row in rows]
    vectors = np.array([row[1:] for row in rows])

    # Close the connection
    conn.close()
    return vectors, identifiers

# Load the pre-trained Word2Vec model (Google's model, for instance)
word2vec_model = api.load('word2vec-google-news-300')


#vectorise the prediction 
def vectorise_prediction(prediction,target_dimension):
    # Get vector for prediction
    prediction_vector = vectorize_text(tokenize(prediction)) 
    #adjusting dimensions of prediction vector and query vector
    prediction_vector = adjust_vector_dimensions(prediction_vector,target_dimension)
    return prediction_vector

############################---LSH using faiss-----################################### 
#get all the query vectors needed for LSH
def get_vectors_for_query(queries,vector_dimention):
    #get query vector
    query_vectors = []
    for query in queries:
        vector = vectorize_text(tokenize(query))
        query_vector = adjust_vector_dimensions(vector,vector_dimention)
        query_vectors.append(query_vector)
    return query_vectors


def LSH(query_vector, database_vectors, identifiers):
    
    
    # Create an index with Faiss
    d = database_vectors.shape[1]  # Dimension of your vectors
    index = faiss.IndexFlatL2(d)  # Using an L2 distance index
    index.add(database_vectors)  # Add your vectors to the index

    # Querying with Faiss
    k = 1  # Number of nearest neighbors to retrieve
    
    # Searching for nearest neighbors using Faiss
    D, I = index.search(np.array([query_vector]), k)
    print("Indices of nearest neighbors:", I)
    print("Distances to nearest neighbors:", D)


    # Retrieve identifiers of nearest neighbors
    retrieved_identifiers = [identifiers[idx] for idx in I[0]]

    print("Identifiers of nearest neighbors:", retrieved_identifiers)

    # Retrieve identifiers of nearest neighbors
    retrieved_identifier = identifiers[I[0][0]]

    # Retrieve the vector for the identified movie/document
    movie_vector = database_vectors[I[0][0]]

    return movie_vector, retrieved_identifier  #movie_vector and ttconst


#cosine_sim = compute_cosine_similarity(prediction_vector,movie_vector)
       # print("The retrieved movie vector: ")
        #print(movie_vector)
       # print()
        #print("The cosine similarity between the movie vector and the prediction is: ", cosine_sim)"""
