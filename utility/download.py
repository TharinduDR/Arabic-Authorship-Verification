from google_drive_downloader import GoogleDriveDownloader as gdd
import os


def download_from_google_drive(file_id, file_name):
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path="data" + "/" + file_name)