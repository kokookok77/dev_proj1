import os
import json
import random
import collections
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from PIL import Image
import numpy as np
import zipfile
 
import config


# 일부 손상된 이미지에 대해서도 읽을 수 있게 해줌
Image.LOAD_TRUNCATED_IMAGES = True


# data set category
# test with small set
category = {"nature-scene": ["beach"],
            "person-made": ["tower"]}
# category = {"nature-scene": ["beach", "cave", "island", "lake", "mountain"],
#             "person-made": ["amusement park", "palace", "park", "restaurant", "tower"]}


# feature extraction model load
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

class KoreaTouristSpotDataSet(Sequence):
    """
    usage 
    data_set = KoreaTouristSpotDataSet(anno_type="caption", mode="train") or 
    anno_type in {"caption", "hashtag"}, mode in {"train", "valid", "test", "total"}
    data_set.load_data() 
    """

    def __init__(self, anno_type="caption", mode="total"):
        super().__init__()
        self.BATCH_SIZE = config.BATCH_SIZE
        self.BUFFER_SIZE = config.BUFFER_SIZE
        self.embedding_dim = config.embedding_dim
        self.units = config.units
        self.top_k = config.top_k
        self.vocab_size = self.top_k + 1
        # Shape of the vector extracted from InceptionV3 is (64, 2048)
        # These two variables represent that vector shape
        self.features_shape = config.features_shape
        self.attention_features_shape = config.attention_features_shape
        self.anno_type = anno_type
        self.mode = mode

        # 파일 없을 경우 구글 드라이브에서 다운
        if not os.path.exists("./data/Korean-Tourist-Spot-Dataset-v0.1.zip"):
            print("data set is not ready. Start Download..")
            print("download complete")
            pass
        
        # zip파일 압축해제 안된 경우 압축 해제
        if not os.path.exists("./data/DGU-AI-LAB-Korean-Tourist-Spot-Dataset-62a483a"):
            print("zip file is ready but not uncompressed.")
            print("start uncompress..")
            try:
                with zipfile.ZipFile("./data/Korean-Tourist-Spot-Dataset-v0.1.zip") as zf:
                    zf.extractall("./data/")
                    print("uncompress success")
            except:
                print("uncompress fail")

        else:
            image_path_to_caption = collections.defaultdict(list)
            
            caption_vector = []
            img_name_vector = []

            for key, val in category.items():
                for v in val:
                    json_dir = os.path.join("./data/DGU-AI-LAB-Korean-Tourist-Spot-Dataset-62a483a/kts//" + self.mode, key, v, v + ".json")
                    image_folder_dir = os.path.join(
                        "./data/DGU-AI-LAB-Korean-Tourist-Spot-Dataset-62a483a/kts//" + self.mode, key, v, "images")

                    with open(json_dir, encoding="UTF-16") as f:
                        js = json.load(f)
                    f.close()

                    for j in js:
                        image_file_dir = os.path.join(
                            image_folder_dir, j["img_name"] + ".jpg")
                        # image = Image.open(image_file_dir)
                        # j["image"]   = image
                        if self.anno_type=="caption":
                            image_path_to_caption[image_file_dir] = f"<start> {j['text']} <end>"
                        else:
                            image_path_to_caption[image_file_dir] = f"<start> {j['hashtag']} <end>"

                    image_paths = list(image_path_to_caption.keys())
                    
                    random.shuffle(image_paths)


            for image_path in image_paths:
                #append 는 단순 요소 추가, extend는 iterable 요소 모두 추가
                caption = image_path_to_caption[image_path]
                caption_vector.append(caption)
                # caption_vector.extend(caption)
                # image 한 장에 caption이 다수 일때
                img_name_vector.append(image_path)

            print("total Set : ", len(caption_vector))

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


            # Choose the top 5000 words from the vocabulary
            tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.top_k,
                                                            oov_token="<unk>",
                                                            filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
            tokenizer.fit_on_texts(caption_vector)
            train_seqs = tokenizer.texts_to_sequences(caption_vector)

            tokenizer.word_index['<pad>'] = 0
            tokenizer.index_word[0] = '<pad>'

            # Pad each vector to the max_length of the captions
            # If you do not provide a max_length value, pad_sequences calculates it automatically
            cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')


            # Calculates the max_length, which is used to store the attention weights
            max_length = calc_max_length(train_seqs)
            

            img_to_cap_vector = collections.defaultdict(list)
            for img, cap in zip(img_name_vector, cap_vector):
                img_to_cap_vector[img].append(cap)

            # Create training and validation sets using an 80-20 split randomly.
            img_keys = list(img_to_cap_vector.keys())
            random.shuffle(img_keys)

            slice_index = int(len(img_keys)*0.8)
            img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

            img_name_train = []
            cap_train = []
            for imgt in img_name_train_keys:
                capt_len = len(img_to_cap_vector[imgt])
                img_name_train.extend([imgt] * capt_len)
                cap_train.extend(img_to_cap_vector[imgt])

            print("train set : ", len(cap_train))

            img_name_val = []
            cap_val = []
            for imgv in img_name_val_keys:
                capv_len = len(img_to_cap_vector[imgv])
                img_name_val.extend([imgv] * capv_len)
                cap_val.extend(img_to_cap_vector[imgv])

            print("validation set : ", len(cap_val))

            dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

            # Use map to load the numpy files in parallel
            dataset = dataset.map(lambda item1, item2: tf.numpy_function(
                    map_func, [item1, item2], [tf.float32, tf.int32]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

            # Shuffle and batch
            dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            print("data set load complete!")
            return dataset, img_name_val, cap_val, tokenizer

def load_image(image_path):
    # 손상된 이미지 읽을 때 에러남
    # img = tf.io.read_file(image_path)
    # img = tf.image.decode_jpeg(img, channels=3)
    
    # PIL로 손상된 이미지도 읽을 수 있음 세부설정은 import 문 아래 참고
    # image_path 는 tensor string으로 주어지기 때문에 캐스팅 필요 안할경우 에러
    image_path = image_path.numpy().decode('utf-8')
    img = Image.open(image_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap

# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)
