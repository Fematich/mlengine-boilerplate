#!/usr/bin/python
PROJECT_ID = '<your-project>'
BUCKET_NAME = '<your-bucket>'

DATA_DIR = 'gs://{}/data/'.format(BUCKET_NAME)
TFRECORD_DIR = 'gs://{}/tfrecords-all/'.format(BUCKET_NAME)
MODEL_DIR = 'gs://{}/model/'.format(BUCKET_NAME)
MODEL_NAME = 'flowers'
WIDTH = 50
HEIGHT = 50
NUM_LABELS = 5
FEAT_LEN = WIDTH*HEIGHT*3
BATCH_SIZE = 64
