import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium") # 'gpt2-medium' is one of the available GPT-2 models. You can also use 'gpt2', 'gpt2-large', or 'gpt2-xl'.
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.eval()  # Set the model to evaluation mode

def generate_response(context, user_query, max_length=100):
    """
    context: str, the top-k movie entries or descriptions retrieved.
    user_query: str, the original query from the user.
    max_length: int, maximum length of the generated response.
    """
    # Combine the context and user query
    input_text = context + user_query
    
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate a response from the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output to get the generated text
    generated_text = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return generated_text

context = "Movie 1: Inception - A mind-bending thriller by Christopher Nolan. Movie 2: Interstellar - A journey through space and time. "
user_query = "Can you suggest a movie similar to Harry Potter released between 2000 and 2010 in the fantasy genre?"
response = generate_response(context, user_query)
print(response)
