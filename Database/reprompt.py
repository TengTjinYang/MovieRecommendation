import pandas as pd

# File paths
name_path = 'test/ImdbNameTest.csv'
basics_path = 'test/ImdbTitleBasicsTest.csv'
principals_path = 'test/ImdbTitlePrincipalsTest.csv'
ratings_path = 'test/ImdbTitleRatingsTest.csv'

# Load the files into pandas DataFrames
name_df = pd.read_csv(name_path)
basics_df = pd.read_csv(basics_path)
principals_df = pd.read_csv(principals_path)
ratings_df = pd.read_csv(ratings_path)

def generate_query(tconst):
    # Extract the row with the given 'tconst'
    movie_basics_df = basics_df[basics_df['tconst'] == tconst]
    movie_principals_df = principals_df[principals_df['tconst'] == tconst]
    movie_ratings_df = ratings_df[ratings_df['tconst'] == tconst]

    # Convert DataFrames to Series for single-row DataFrames
    movie_basics_row = movie_basics_df.iloc[0]
    movie_ratings_row = movie_ratings_df.iloc[0]

    # Build query components
    query_components = []
    
    # Title
    title = movie_basics_row['primaryTitle'] if not pd.isna(movie_basics_row['primaryTitle']) else movie_basics_row['originalTitle']
    if title:
        query_components.append(f"Can you provide details for the movie '{title}'")
    
    # Release Year
    if not pd.isna(movie_basics_row['startYear']):
        query_components.append(f"from {movie_basics_row['startYear']}")
    
    # Rating
    if not pd.isna(movie_ratings_row['averageRating']):
        query_components.append(f"which has a rating of {movie_ratings_row['averageRating']}")
    
    # Film Type
    film_type_phrase = "suitable for general audiences" if movie_basics_row['isAdult'] == 0 else "an adult film"
    query_components.append(f"is {film_type_phrase}")
    
    # Personnel
    personnel_names = []
    for _, row in movie_principals_df.iterrows():
        if '["Self"]' in row['characters']:
            nconst = row['nconst']
            person_info_df = name_df[name_df['nconst'] == nconst]
            if not person_info_df.empty:
                person_info = person_info_df.iloc[0]
                personnel_names.append(f"{person_info['primaryName']} as themselves")
    
    if personnel_names:
        personnel_string = '; '.join(personnel_names)
        query_components.append(f"and features {personnel_string}")

    # Combine the components to form the final query
    query_filled = ', '.join(query_components) + "."
    return query_filled

# Example usage
tconst_value = 'tt0816692'  # Replace with your actual tconst value
query = generate_query(tconst_value)
print(query)