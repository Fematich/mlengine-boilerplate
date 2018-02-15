#!/usr/bin/python
PROJECT_ID = '<your-project>'

DATA_DIR = 'gs://<your-bucket>/data/'
TFRECORD_DIR = 'gs://<your-bucket>/tfrecords-all/'
MODEL_DIR = 'gs://<your-bucket>/model/'
MODEL_NAME = 'flowers'
WIDTH = 50
HEIGHT = 50
NUM_LABELS = 5
FEAT_LEN = WIDTH*HEIGHT*3
BATCH_SIZE = 64
