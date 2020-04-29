from algo.run_experiment import run_use_experiment, run_laser_experiment, run_multilingual_sbert_experiment
from examples.maimonides.config import book_1_google_drive, book_1_id, file_1_name, book_2_google_drive, book_2_id, \
    file_2_name
from utility.download import download_from_google_drive
from utility.reader import read_book
import os

if book_1_google_drive:
    book_1 = download_from_google_drive(book_1_id, file_name=file_1_name)


if book_2_google_drive:
    book_2 = download_from_google_drive(book_2_id, file_name=file_2_name)


list_1 = read_book(os.path.join("data", file_1_name))
list_2 = read_book(os.path.join("data", file_2_name))

#run_use_experiment(list_1, list_2, "result_0.8.txt", run_sts_experiment=True, threshold=0.8, optimize=False)
#run_laser_experiment(list_1, list_2, "laser_result_0.80.txt", run_sts_experiment=True, threshold=0.80, max_no=5)

run_multilingual_sbert_experiment(list_1, list_2, "sbert_result_0.80.txt", run_sts_experiment=True, threshold=0.80)