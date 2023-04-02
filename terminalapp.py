import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
import json
from transformers.utils import logging

logging.set_verbosity(40)


userchats=[]
sentiment=[0,0,0,0]
model1= tf.keras.models.load_model("C:/Users/shukl/Desktop/Song recommendor/model")


class MakeApiCall:

    def get_data(self, api):
        response = requests.get(f"{api}")
        if response.status_code == 200:

            self.formatted_print(response.json())
        else:
            print(
                f"Hello person, there's a {response.status_code} error with your request")


    def formatted_print(self, obj):
       
        print("Hey you should try listening to "+obj["tracks"]["track"][0]['name']+ " by "+obj["tracks"]["track"][0]['artist']['name'])


    def __init__(self, api):
        self.get_data(api)


def predict(stringa):
    with open("C:/Users/shukl/Desktop/Song recommendor/tokenizer.pickle", 'rb') as handle:
      tokenizer = pickle.load(handle)
      list_tokenized_train = tokenizer.texts_to_sequences([stringa])
      x_train = pad_sequences(list_tokenized_train, maxlen=100)
      output=model1.predict(x_train,verbose=0)
      return output

def song():

  sum=0
  for i in userchats:
    prediction=predict(i)
    sentiment[0]+=prediction[0][0]
    sentiment[1]+=prediction[0][1]
    sentiment[2]+=prediction[0][2]
    sentiment[3]+=prediction[0][3]

  for i in sentiment:
    sum+=i
  
  for i in range(0,4):
    sentiment[i]/=sum

  sum=sentiment[0]
  index=0
  for i in range(0,4):
    if(sum<sentiment[i]):
      sum=sentiment[i]
      index=i
  
 

  if  index == 0:
     MakeApiCall("http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag=metal&api_key=448cbd3787045393320ae45027e0da94&format=json&limit=5")
  if  index == 1:
     MakeApiCall("http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag=lofi&api_key=448cbd3787045393320ae45027e0da94&format=json&limit=5")
  if  index == 2:
     MakeApiCall("http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag=dance&api_key=448cbd3787045393320ae45027e0da94&format=json&limit=5")
  if  index == 3:
     MakeApiCall("http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag=sad&api_key=448cbd3787045393320ae45027e0da94&format=json&limit=5")


def load_tokenizer_and_model(model="microsoft/DialoGPT-medium"):
  """
    Load tokenizer and model instance for some specific DialoGPT model.
  """
  # Initialize tokenizer and model
  print("Loading model...")
  tokenizer = AutoTokenizer.from_pretrained(model)
  model = AutoModelForCausalLM.from_pretrained(model)
  tokenizer.padding_side='left'
  # Return tokenizer and model
  return tokenizer, model


def generate_response(chat_history_ids,tokenizer, model, chat_round):
  """
    Generate a response to some user input.
  """

  x=input(">> You:")
  # Encode user input and End-of-String (EOS) token
  new_input_ids = tokenizer.encode(  x+tokenizer.eos_token, return_tensors='pt')

  bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_round > 0 else new_input_ids

  chat_history_ids = model.generate(bot_input_ids, max_length=30000, pad_token_id=tokenizer.eos_token_id)
  
  # Print response
  response=str(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
  print("DialoGPT:",response)
  userchats.append(x)

  # Return the chat history ids
  return chat_history_ids,response



def chat_for_n_rounds(chat_round=0):
  """
  Chat with chatbot for n rounds (n = 5 by default)
  """  
  # Initialize history variable
  tokenizer, model = load_tokenizer_and_model()
  chat_history_ids = None
  # Chat for n rounds
  for i in range(0,10):
    chat_history_ids,response=generate_response(chat_history_ids,tokenizer, model, chat_round)
    chat_round+=1
    if(chat_round%5==0):
      song()


chat_for_n_rounds(0)
