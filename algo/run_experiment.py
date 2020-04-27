import logging
from random import shuffle
import tensorflow_text
import scipy.spatial
import tensorflow_hub as hub
import os

from tqdm import tqdm

from preprocess.cleaning import clean_arabic
from utility.split import batch


def run_use_experiment(list_1, list_2, result_file, batch_size=8, optimize=False, cleaning=True, random_seed=777):
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

        for x in batch(cleaned_list_1, batch_size):
            list_1_embeddings.append(embed(x))

        print("Length of the list 1 embeddings {}".format(str(len(list_1_embeddings))))

        for x in batch(cleaned_list_2, batch_size):
            list_2_embeddings.append(embed(x))

        print("Length of the list 2 embeddings {}".format(str(len(list_2_embeddings))))

    else:
        for x in batch(list_1, batch_size):
            list_1_embeddings.append(embed(x))
        print("Length of the list 1 embeddings {}".format(str(len(list_1_embeddings))))

        for x in batch(list_2, batch_size):
            list_2_embeddings.append(embed(x))

        print("Length of the list 2 embeddings {}".format(str(len(list_2_embeddings))))

    closest_n = 5
    for text, embedding in tqdm(zip(list_1, list_1_embeddings)):
        lines = []
        distances = scipy.spatial.distance.cdist([embedding], list_2_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        lines.append("\n\n======================\n\n")
        lines.append(text)
        lines.append("*************************")

        for idx, distance in results[0:closest_n]:
            if distance < 0.2:
                lines.append(list_2[idx].strip())
                print(1-distance)

            if len(lines) > 3:
                print(lines)
                with open(result_file, mode='a', encoding='utf-8') as result_file:
                    result_file.write('\n'.join(lines))

