#!/usr/bin/env python
# coding: utf-8

# # KTS(Korean Tourist Spot) Dataset Load

# # Class Definition

# In[1]:


#image, label, likes, text, hashtag
category = {"nature-scene": ["beach", "cave", "island", "lake", "mountain"],
            "person-made": ["amusement park", "palace", "park", "restaurant", "tower"]}


# # Load Total Raw Heterogeneous Data(text - json & images - jpg)

# In[2]:


import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import collections
import random
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

file_root_path = "C:/Users/InKwang Oh/Downloads/Korean-Tourist-Spot-Dataset-v0.1/DGU-AI-LAB-Korean-Tourist-Spot-Dataset-62a483a"

def load_data(mode, file_root_path):
    if mode != "train" and mode != "valid" and mode != "test" and mode != "total":
        return None
    
    else:        
        data = np.array([]) # initialization
        
        for key, val in category.items():
            for v in val:
                json_dir = os.path.join(file_root_path, "kts/" + mode, key, v, v + ".json")
                image_folder_dir = os.path.join(file_root_path, "kts/" + mode, key, v, "images")

                with open(json_dir, encoding="UTF-16") as f:
                    js = json.load(f)
                f.close()

                for j in js:
                    image_file_dir = os.path.join(image_folder_dir, j["img_name"] + ".jpg")
                    image = image_file_dir
                    j["image"]   = image   
                    
                js = np.asarray(js)
                data = np.append(data, js)
        return data

data = load_data("train", file_root_path = file_root_path)


# In[10]:


# Group all captions together having the same image ID.
image_path_to_caption = collections.defaultdict(list)
for idx, val in enumerate(data):
    caption = f"<start> {val['text']} <end>"
    image_path = val['image']
    image_path_to_caption[image_path].append(caption)


# In[141]:


image_paths = list(image_path_to_caption.keys())
# random.shuffle(image_paths)

# Select the first 6000 image_paths from the shuffled set.
# Approximately each image id has 5 captions associated with it, so that will 
# lead to 100 examples.
train_image_paths = image_paths


# In[142]:


train_captions = []
img_name_vector = []

for image_path in train_image_paths:
  caption_list = image_path_to_caption[image_path]
  train_captions.extend(caption_list)
  img_name_vector.extend([image_path] * len(caption_list))


# In[143]:


print(train_captions[19])
Image.open(img_name_vector[19])


# # InceptionV3를 사용하여 이미지 전처리

# In[144]:


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# In[145]:


image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# In[123]:


# Get unique images
encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

for img, path in image_dataset:
  batch_features = image_features_extract_model(img)
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())


# In[146]:


# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# In[147]:


# Choose the top 5000 words from the vocabulary
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)


# In[148]:


for i in range(5) :
    print(train_captions[i], end="\n")

for i in range(5) :
    print(train_seqs[i], end="\n")


# In[149]:


tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'


# In[150]:


# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')


# In[151]:


# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)


# In[152]:


img_name_train = img_name_vector.copy()
cap_train = cap_vector.copy() 


# # 학습용 tf.data 데이터 세트 만들기

# In[137]:


# Feel free to change these parameters according to your system's configuration

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64


# In[155]:


# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap


# In[156]:


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# In[162]:


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))


# In[165]:


dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# In[167]:


dataset

