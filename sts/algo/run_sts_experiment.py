import logging
from random import shuffle
import tensorflow_text
import tensorflow_hub as hub
import os
import numpy as np
from laserembeddings import Laser
from numpy import dot
from scipy.stats import pearsonr, spearmanr

from tqdm import tqdm
from numpy.linalg import norm

from preprocess.cleaning import clean_arabic
from sts.reaader import concatenate
from utility.download import download_sts_data_from_google_drive
from utility.split import batch


def run_use_sts_experiment(optimize, cleaning, batch_size=8, random_seed=777):

    df = concatenate("sts_data")

    list_1 = df['text_a'].tolist()
    list_2 = df['text_b'].tolist()

    if optimize:
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

    else:
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

    list_1_embeddings = []
    list_2_embeddings = []

    if cleaning:
        cleaned_list_1 = [clean_arabic(item) for item in list_1]
        cleaned_list_2 = [clean_arabic(item) for item in list_2]

        for x in tqdm(batch(cleaned_list_1, batch_size)):
            list_1_embeddings.extend(embed(x))

        print("Length of the list 1 embeddings {}".format(str(len(list_1_embeddings))))

        for x in tqdm(batch(cleaned_list_2, batch_size)):
            list_2_embeddings.extend(embed(x))

        print("Length of the list 2 embeddings {}".format(str(len(list_2_embeddings))))

    else:
        for x in tqdm(batch(list_1, batch_size)):
            list_1_embeddings.extend(embed(x))
        print("Length of the list 1 embeddings {}".format(str(len(list_1_embeddings))))

        for x in tqdm(batch(list_2, batch_size)):
            list_2_embeddings.extend(embed(x))

        print("Length of the list 2 embeddings {}".format(str(len(list_2_embeddings))))

    predicted_similrities = []
    similarities = df['labels'].tolist()

    for embedding_1, embedding_2 in tqdm(zip(list_1_embeddings, list_2_embeddings)):
        cos_sim = np.dot(embedding_1, embedding_2) / (norm(embedding_1) * norm(embedding_2))
        predicted_similrities.append(cos_sim)

    print("Pearson Coorelation - {}".format(str(pearsonr(similarities, predicted_similrities)[0])))


def run_laser_sts_experiment(cleaning, batch_size=8, random_seed=777):

    df = concatenate("sts_data")

    list_1 = df['text_a'].tolist()
    list_2 = df['text_b'].tolist()

    list_1_embeddings = []
    list_2_embeddings = []

    laser = Laser()

    if cleaning:
        cleaned_list_1 = [clean_arabic(item) for item in list_1]
        cleaned_list_2 = [clean_arabic(item) for item in list_2]

        for x in tqdm(batch(cleaned_list_1, batch_size)):
            list_1_embeddings.extend(laser.embed_sentences(x, lang='ar'))

        print("Length of the list 1 embeddings {}".format(str(len(list_1_embeddings))))

        for x in tqdm(batch(cleaned_list_2, batch_size)):
            list_2_embeddings.extend(laser.embed_sentences(x, lang='ar'))

        print("Length of the list 2 embeddings {}".format(str(len(list_2_embeddings))))

    else:
        for x in tqdm(batch(list_1, batch_size)):
            list_1_embeddings.extend(laser.embed_sentences(x, lang='ar'))
        print("Length of the list 1 embeddings {}".format(str(len(list_1_embeddings))))

        for x in tqdm(batch(list_2, batch_size)):
            list_2_embeddings.extend(laser.embed_sentences(x, lang='ar'))

        print("Length of the list 2 embeddings {}".format(str(len(list_2_embeddings))))

    predicted_similrities = []
    similarities = df['labels'].tolist()

    for embedding_1, embedding_2 in tqdm(zip(list_1_embeddings, list_2_embeddings)):
        cos_sim = dot(embedding_1, embedding_2) / (norm(embedding_1) * norm(embedding_2))
        predicted_similrities.append(cos_sim)

    print("Pearson Coorelation - {}".format(str(pearsonr(similarities, predicted_similrities)[0])))



