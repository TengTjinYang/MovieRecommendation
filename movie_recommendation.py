import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.eval()  # Set the model to evaluation mode

def generate_response(user_query, max_length=100):
    """
    context: str, the top-k movie entries or descriptions retrieved.
    user_query: str, the original query from the user.
    max_length: int, maximum length of the generated response.
    """
    
    # Tokenize the input text
    input_ids = tokenizer.encode(user_query, return_tensors="pt")
    
    # Generate a response from the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output to get the generated text
    generated_text = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return generated_text

user_query = "Can you suggest a movie released between 2000 and 2010 with Chris Hemsworth in it?"
response = generate_response(user_query)
print(response)
