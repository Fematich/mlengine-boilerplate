#!/usr/bin/python
PROJECT_ID = 'project-id'
BUCKET = 'gs://bucket/'
DATA_DIR = BUCKET + 'data/'
TFRECORD_DIR = BUCKET + 'tfrecords/'
MODEL_DIR = BUCKET + 'model/'
MODEL_NAME = 'flowers'
WIDTH = 50
HEIGHT = 50
NUM_LABELS = 5
FEAT_LEN = WIDTH*HEIGHT*3
BATCH_SIZE = 64