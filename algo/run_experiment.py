import logging
from random import shuffle
import tensorflow_text
import tensorflow_hub as hub
import os
import numpy as np
from laserembeddings import Laser

from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm

from preprocess.cleaning import clean_arabic
from utility.split import batch


def run_use_experiment(list_1, list_2, result_file, batch_size=8, threshold=0.8, optimize=False, cleaning=True, random_seed=777):
    shuffle(list_1)
    shuffle(list_2)

    if os.path.isfile(result_file):
        os.remove(result_file)

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

    for text, embedding in tqdm(zip(list_1, list_1_embeddings)):
        lines = []
        lines.append("\n")
        lines.append("=========================")
        lines.append(text)
        lines.append("*************************")

        selected_text = []
        similarity = []

        for duplicate_text, duplicate_embedding in zip(list_2, list_2_embeddings):
            cos_sim = dot(embedding, duplicate_embedding) / (norm(embedding) * norm(duplicate_embedding))
            if cos_sim > threshold:
                selected_text.append(duplicate_text)
                similarity.append(cos_sim)

        if len(selected_text) > 0:
            results = zip(selected_text, similarity)
            results = sorted(results, key=lambda x: -x[1])

            for similar_text, similarity in results:
                lines.append(similar_text)
                print(similarity)

            with open(result_file, mode='a', encoding='utf-8') as file:
                file.write('\n'.join(lines))


def run_laser_experiment(list_1, list_2, result_file, batch_size=8, threshold=0.8, cleaning=True, random_seed=777):
    shuffle(list_1)
    shuffle(list_2)

    if os.path.isfile(result_file):
        os.remove(result_file)

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

    for text, embedding in tqdm(zip(list_1, list_1_embeddings)):
        lines = []
        lines.append("\n")
        lines.append("=========================")
        lines.append(text)
        lines.append("*************************")

        selected_text = []
        similarity = []

        for duplicate_text, duplicate_embedding in zip(list_2, list_2_embeddings):
            cos_sim = dot(embedding, duplicate_embedding) / (norm(embedding) * norm(duplicate_embedding))
            if cos_sim > threshold:
                selected_text.append(duplicate_text)
                similarity.append(cos_sim)

        if len(selected_text) > 0:
            results = zip(selected_text, similarity)
            results = sorted(results, key=lambda x: -x[1])

            for similar_text, similarity in results:
                lines.append(similar_text)
                print(similarity)

            with open(result_file, mode='a', encoding='utf-8') as file:
                file.write('\n'.join(lines))



