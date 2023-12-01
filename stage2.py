# -*- coding: utf-8 -*-
"""Stage 2 of My LLM Load and Inference for IR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fqx8LBmP4K7dy29ynEoGQ4bgVmrLFIJr
"""

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

#pip install accelerate
#pip install bitsandbytes
#pip install transformers
#pip install SentencePiece
#pip install evaluate
#pip install bert_score

import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
import accelerate
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import LlamaTokenizer

load_in_8bit = False
model_name_l = "lmsys/vicuna-7b-v1.5"

model_l = AutoModelForCausalLM.from_pretrained(
    model_name_l,
    torch_dtype=torch.float16,
    load_in_8bit=load_in_8bit,
    device_map="auto"
)
tokenizer_l = LlamaTokenizer.from_pretrained(model_name_l)

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

ground_truth = [
    "Notting Hill or Pretty Woman",
    "Blade Runner 2049, Interstellar",
    "Wonder Woman", "Captain Marvel",
    "The Prestige", "Donnie Darko",
    "A Quiet Place", "The Witch",
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

def response_gen(input_ids):
  temperature=0.7
  with torch.no_grad():
      generation_output = model_l.generate(
          input_ids=input_ids,
          temperature=temperature,
          top_p = 1.0,
          do_sample=True,
          return_dict_in_generate=True,
          max_new_tokens=100,
      )
  s = generation_output.sequences[0][len(input_ids[0]):]
  output = tokenizer_l.decode(s)
  return output

predictions = []
for question in questions:
  input = tokenizer_l(question, return_tensors='pt')
  input_ids = input["input_ids"].to("cuda")

  output = response_gen(input_ids)
  print("Prediction:", output)
  predictions.append(output)