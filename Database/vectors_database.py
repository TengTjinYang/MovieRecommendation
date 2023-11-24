import yang_vectorize
import database_creation
import pandas as pd
from sqlalchemy import create_engine

# Access the 'vectors_df' from each script
df1 = yang_vectorize.vectors_df
df2 = database_creation.vectors_df

# Merge the dataframes on 'tconst'
merged_df = pd.merge(df1, df2, on='tconst', how='inner')

# Connect to SQL database
engine = create_engine('sqlite:///movie_data.db')

# Export to SQL
merged_df.to_sql('vectors', engine, if_exists='replace', index=False)

with engine.connect() as connection:
    queried_df = pd.read_sql('SELECT * FROM vectors LIMIT 5', con=connection)

# Display the queried rows
print(queried_df.head())