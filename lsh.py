from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np

#tokenise text
def tokenize(text):
    return text.lower().split()

# convert the llm output into vector
def vetorize_text(text):
    valid_words = [word for word in text if word in word2vec_model.key_to_index]
    if len(valid_words) > 0:
        return np.mean(word2vec_model[valid_words], axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)


#need to use padding on the vectors so they match with the dimentions of our dataset vectors    

# lsh the original query to our dataset - it should return a movie name
# convert the lsh output to vector
# compare the two vectors (lsh output and llm output) using cosine similarity


# -----------------------------------------  main body  ----------------------------------------#

#lsh class
class LSH:
    def __init__(self, n_bits, n_planes):
        self.n_bits = n_bits
        self.n_planes = n_planes
        self.hyperplanes = np.random.randn(n_planes, n_bits)
        self.buckets = {}

    def hash_vector(self, vec):
        return ''.join(['1' if np.dot(self.hyperplanes[i], vec) >= 0 else '0' for i in range(self.n_planes)])

    def add_vector_to_bucket(self, vec, idx):
        hashed_vec = self.hash_vector(vec)
        if hashed_vec not in self.buckets:
            self.buckets[hashed_vec] = []
        self.buckets[hashed_vec].append(idx)

    def query(self, query_vec):
        hashed_query = self.hash_vector(query_vec)
        return self.buckets.get(hashed_query, [])


# Sample questions and predictions
questions = [
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

predictions = ['obviously, when Harry met Sally.',
               'nobody has answered this question yet.',
               'everyone will know who she is.',
               'everybody!',
               "everyone talks about it but you haven't seen it. What is it?",
               'nobody likes me',
               '</s>',
               '</s>',
               '</s>',
               "nobody wants to watch a movie about time travel that doesn't have a good plot or characters. Here are a few movies that deal with time travel that I've seen:1. Back to the Future (1985) 2. The Time Traveler's Wife (2009) 3. 12 Monkeys (1995) 4. Bill & Ted's Excellent Adventure (1",
               'somebody help me out here!</s>',
               '</s>',
               '</s>',
               'држава I recommend "The Girl on the Train" by Paula Hawkins. It\'s a suspense thriller with an unexpected twist that will keep you on the edge of your seat. The story follows a woman who becomes embroiled in a mystery when she witnesses a disturbing act on a commuter train. As she delves deeper into the investigation, she uncovers dark secrets that will change her life forever. The plot is well-crafted',
               '</s>',
               '</s>',
               '</s>',
               '̶There are many movies with exceptional soundtracks, but one that stands out is "The Grand Budapest Hotel" directed by Wes Anderson. The film\'s score, composed by Alexandre Desplat, is a beautiful blend of classical and jazz music that perfectly captures the whimsical and adventurous spirit of the story. The soundtrack also features a range of other classical and pop songs that add to the film\'s charm and nostalgia. Overall',
               '</s>',
               'everybody knows this is nowhere by chuck palahniuk</s>']

print(len(questions))
print(len(predictions))


# Load the pre-trained Word2Vec model (Google's model, for instance)
word2vec_model = api.load('word2vec-google-news-300')


#tokenize prediction
tokenized_text = tokenize(predictions[0])
# Get vector for prediction
prediction_vectors = vetorize_text(tokenized_text) #[get_text_vector(text) for text in tokenized_text[:len(questions)]]
#prediction_vectors = [get_text_vector(text) for text in tokenized_text[len(questions):]]

print(prediction_vectors)

#####some stuff for lsh
# Example dataset (randomly generated)
data = np.random.rand(10, 5)  # 10 vectors of dimension 5

lsh = LSH(n_bits=4, n_planes=3)

# Adding vectors to buckets
for idx, vector in enumerate(data):
    lsh.add_vector_to_bucket(vector, idx)

# Querying for a vector
query_vector = np.random.rand(5)
query_result = lsh.query(query_vector)
print("Query result:", query_result)