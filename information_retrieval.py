import lsh

#getting database vectors
print("Connecting to database...")
database_vectors, identifiers = lsh.load_database_vectors('movie_data.db')

#test_queries
test_queries = [
    'Recommend a classic romantic comedy from the 1990s.',
    'What are some award-winning science fiction movies?',
    'Recommend a movie with an iconic female superhero.',
    'Name a movie that features a mind-bending narrative like Inception.',
    'Name a recent horror movie that has gained critical acclaim.',
    'What is a popular animated movie suitable for all ages?',
    'What are some iconic action movies from the 1980s?',
    'Can you suggest a movie that is both a comedy and a mystery?',
    'Which movies are known for their stunning visual effects?',
    'Recommend a movie that deals with time travel.',
    'What are some critically acclaimed foreign language films?',
    'Can you recommend a biographical movie about a famous musician?',
    'What are some good movies for children under 10?',
    'Recommend a suspense thriller with an unexpected twist.',
    'Which movies have won the Best Picture Oscar in the last decade?',
    'Can you suggest a movie that focuses on artificial intelligence?',
    'What are some of the best adaptations of comic books to movies?',
    'Recommend a movie that is known for its exceptional soundtrack.',
    'Which movies feature an ensemble cast?',
    'Can you suggest a film that is a great example of film noir?'
]

test_predictions = [
    "everybody loves meg ryan and tom hanks in this movie. It's a classic romantic comedy from the 1990s that features Meg Ryan and Tom Hanks in lead roles. The movie revolves around the story of a woman who is in love with a man who is already engaged to someone else. The woman finds out that the man is about to marry the wrong person and decides to do everything she can to stop the wedding from taking place.", 
    "nobody knows Is there a list of award-winning science fiction movies?</s>", 
    "hopefully one that is not too old and not too recent. My choices are: 1. Wonder Woman (2009) 2. Black Widow (2021) 3. Captain Marvel (2019) 4. The Incredibles (2004) Which one would you recommend and why?</s>",
    "nobody can predict what's gonna happen next</s>",
    "nobody can name a recent horror movie that has gained critical acclaim because the horror genre is generally not well-regarded by critics, who tend to focus on more 'prestigious' genres like drama and comedy. However, there are some horror movies that have been well-received by critics, such as 'The Blair Witch Project' and 'Paranormal Activity.' These movies were praised for their innovative use of found footage and their ability to create",
    "</s>", 
    "everybody knows the classics like 'Rambo: First Blood Part II' and 'Commando', but what are some other iconic action movies from the 1980s that you might not have heard of? Here are 10 that come to mind: 1. 'Road House' (1989) - This movie stars Patrick Swayze as a bouncer who's hired to keep a rowdy bar in line.",
    "</s>",
    "</s>",
    "dealing with the butterfly effect I would recommend the movie 'The Butterfly Effect' (2004), directed by Eric Bress and J. Mackye Gruber. The film follows a young man named Evan Treborn, played by Ashton Kutcher, who discovers that he has the ability to travel through time. However, every time he goes back in time, he inadvertently changes small events that cause a ripple effect, ultimately",
    "nobody knows. I can only suggest some of my favorite films in different languages. 1. 'The Seventh Seal' (1957) directed by Ingmar Bergman, a Swedish film that deals with existential themes and is considered a masterpiece of world cinema. 2. 'Breathless' (1960) directed by Jean-Luc Godard, a French New Wave film that is known for its fast-",
    "</s>",
    "</s>",
    "hopefully one that's not too predictable I recently read 'The Girl on the Train' by Paula Hawkins and really enjoyed the unexpected twist at the end. Is there anything similar?</s>",
    "</s>", 
    "obviously i am not looking for any of the recent marvel movies or the like, i am looking for something more thought-provoking.</s>", 
    "obviously, The Dark Knight Trilogy, Spider-Man, and The Avengers come to mind, but are there any other great adaptations that people might not think of?</s>",
    "nobody knows me</s>",
    "</s>",
    "</s>" ]

ground_truth = [
    "Notting Hill or Pretty Woman",
    "Blade Runner 2049, Interstellar",
    "Wonder Woman, Captain Marvel",
    "The Prestige, Donnie Darko",
    "A Quiet Place, The Witch",
    "Toy Story, Frozen",
    "Die Hard, The Terminator",
    "Knives Out, Clue",
    "Avatar, Gravity",
    "Back to the Future, Looper",
    "Parasite, Amélie",
    "Bohemian Rhapsody, Ray",
    "Finding Nemo, The Lion King",
    "The Sixth Sense, Gone Girl",
    "The Shape of Water, Nomadland",
    "Ex Machina, A.I. Artificial Intelligence",
    "The Dark Knight, Spider-Man: Into the Spider-Verse",
    "La La Land, Guardians of the Galaxy",
    "Ocean's Eleven, The Grand Budapest Hotel",
    "Chinatown, The Maltese Falcon"
]

#getting the dimentions of the vectors from the database
target_dimension = database_vectors.shape[1]
print("Target dimension:", target_dimension)

#vectorise all test_predictions
prediction_vectors =[]
for prediction in test_predictions:
    prediction_vectors.append(lsh.vectorise_prediction(prediction, target_dimension))
    

print(len(test_queries))
print(len(test_predictions))

#lsh.load_csv()

option = input("Would you like to enter a query (1) or use the test queries(2): ")
if option == "1":
    user_query = input("Enter Query: ")
    query_vector = lsh.vectorize_text(lsh.tokenize(user_query))
    query_vector = lsh.adjust_vector_dimensions(query_vector,target_dimension)
    retrieved_vector,retrieved_ttconst = lsh.LSH(query_vector,database_vectors,identifiers)
    print(retrieved_ttconst)
    prediction = "Harry meets Sally"
    prediction_vector = lsh.vectorise_prediction(prediction, target_dimension)
    cosine_sim = lsh.compute_cosine_similarity(prediction_vector,retrieved_vector)
    print()
    print("Query: ", user_query)
    print("Prediction: ", prediction)
    print("The cosine similarity between the movie vector and the prediction is: ", cosine_sim)
   # ground_truth = 
    #bert_score = lsh.bert_score(prediction, ground_truth)
    #print("Bert score: ", bert_score)
else: 
    query_vectors = lsh.get_vectors_for_query(test_queries,target_dimension)
    for i in range (len(query_vectors)):
        retrieved_vector,retrieved_ttconst = lsh.LSH(query_vectors[i],database_vectors,identifiers)
        print(retrieved_ttconst)
        print(test_queries[i])
        cosine_sim = lsh.compute_cosine_similarity(prediction_vectors[i],retrieved_vector)
        print()
        print("The cosine similarity between the movie vector and the prediction is: ", cosine_sim)
    bert_score = lsh.bert_score(test_predictions, ground_truth)
    print("Bert score: ", bert_score)
        