import os
import os.path
import pickle
from urllib import urlretrieve

FILEPATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(FILEPATH, "datasets")

ENRON_SPAM_URL = (
    "http://csmining.org/index.php/"
    "enron-spam-datasets.html"
    "?file=tl_files/Project_Datasets/Enron-Spam%20datasets/Preprocessed"
    "/enron1.tar.tar"
)

TAR_FILE_NAME = "enron1.tar.tar"
ENRON_DATA_DIR_NAME = "enron1"

def download_tarfile():
    tarfile_path = os.path.join(DATA_DIR, TAR_FILE_NAME)
    if os.path.isfile(tarfile_path):
        print "Tarfile already downloaded!"
        return

    print "Downloading enron1.tar.tar"
    urlretrieve(ENRON_SPAM_URL, tarfile_path)
    print "Download complete!"

def extract_tarfile():
    tarfile_path = os.path.join(DATA_DIR, TAR_FILE_NAME)
    enron_data_dir = os.path.join(DATA_DIR, ENRON_DATA_DIR_NAME)
    if os.path.isdir(enron_data_dir):
        print "Tarfile already extracted!"
        return

    print "Extracting enron1.tar.tar"
    os.system("tar -xf %s -C %s" % (tarfile_path, DATA_DIR))
    print "Extraction complete!"

if __name__ == "__main__":
    download_tarfile()
    extract_tarfile()
