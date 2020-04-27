from algo.run_experiment import run_use_experiment
from examples.maimonides.config import book_1_google_drive, book_1_id, file_1_name, book_2_google_drive, book_2_id, \
    file_2_name
from utility.download import download_from_google_drive
from utility.reader import read_book

if book_1_google_drive:
    book_1 = download_from_google_drive(book_1_id, path="", file_name=file_1_name)


if book_2_google_drive:
    book_2 = download_from_google_drive(book_2_id, path="", file_name=file_2_name)


list_1 = read_book(file_1_name)
list_2 = read_book(file_2_name)

run_use_experiment(list_1, list_2, "result.txt")