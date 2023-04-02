import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import requests
import json


##File to recommend songs using the previous chats

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
       
      x= "Hey you should try listening to "+obj["tracks"]["track"][0]['name']+ " by "+obj["tracks"]["track"][0]['artist']['name']
      return x


    def __init__(self, api):
        self.get_data(api)


def predict(stringa):
    with open("C:/Users/shukl/Desktop/Song recommendor/tokenizer.pickle", 'rb') as handle:
      tokenizer = pickle.load(handle)
      list_tokenized_train = tokenizer.texts_to_sequences([stringa])
      x_train = pad_sequences(list_tokenized_train, maxlen=100)
      output=model1.predict(x_train,verbose=0)
      return output

def song(userchats):
  sentiment=[0,0,0,0]
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
     return MakeApiCall("http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag=metal&api_key=448cbd3787045393320ae45027e0da94&format=json&limit=5")
  if  index == 1:
     return MakeApiCall("http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag=lofi&api_key=448cbd3787045393320ae45027e0da94&format=json&limit=5")
  if  index == 2:
     return MakeApiCall("http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag=dance&api_key=448cbd3787045393320ae45027e0da94&format=json&limit=5")
  if  index == 3:
     return MakeApiCall("http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag=sad&api_key=448cbd3787045393320ae45027e0da94&format=json&limit=5")
