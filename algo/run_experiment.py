import logging
from random import shuffle
import tensorflow_text
import scipy
import tensorflow_hub as hub
import os

from preprocess.cleaning import clean_arabic


def run_use_experiment(list_1, list_2, result_file, optimize=False, cleaning=True, random_seed=777):
    shuffle(list_1)
    shuffle(list_2)

    if os.path.isfile(result_file):
        os.remove(result_file)

    if optimize:
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

    else:
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

    if cleaning:
        cleaned_list_1 = [clean_arabic(item) for item in list_1]
        list_1_embeddings = embed(cleaned_list_1)
        logging.info("Length of the testing embeddings {}".format(str(len(list_1_embeddings))))
        cleaned_list_2 = [clean_arabic(item) for item in list_2]
        list_2_embeddings = embed(cleaned_list_2)
        logging.info("Length of the testing embeddings {}".format(str(len(list_2_embeddings))))

    else:
        list_1_embeddings = embed(list_1)
        logging.info("Length of the testing embeddings {}".format(str(len(list_1_embeddings))))
        list_2_embeddings = embed(list_2)
        logging.info("Length of the testing embeddings {}".format(str(len(list_2_embeddings))))

    closest_n = 5
    for text, embedding in zip(list_1, list_1_embeddings):
        lines = []
        distances = scipy.spatial.distance.cdist([embedding], list_2_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        if results[0] < 0.2:
            logging.info("\n\n======================\n\n")
            lines.append("\n\n======================\n\n")
            logging.info(text)
            lines.append(text)
            logging.info("*************************")
            lines.append("*************************")

            for idx, distance in results[0:closest_n]:
                if distance < 0.2:
                    logging.info(list_2[idx].strip())
                    lines.append(list_2[idx].strip())
                    logging.info(1-distance)

            with open(result_file, mode='a', encoding='utf-8') as result_file:
                result_file.write('\n'.join(lines))

