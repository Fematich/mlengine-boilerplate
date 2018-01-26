    #!/usr/bin/python
import argparse
import logging
import os
import sys
from datetime import datetime
import numpy as np

import apache_beam as beam
from apache_beam.metrics import Metrics
from tensorflow_transform import coders

from trainer.config import PROJECT_ID, DATA_DIR, TFRECORD_DIR, NUM_LABELS
from trainer.util import schema, read_image

logging.warning('running preprocess')

partition_train = Metrics.counter('partition', 'train')
partition_validation = Metrics.counter('partition', 'validation')
partition_test = Metrics.counter('partition', 'test')
examples_failed = Metrics.counter('build', 'failed')


def build_example((key, label, img_bytes)):
    """Build a dictionary that contains all the features and label to store
    as TFRecord

    Args:
        raw_in: raw data to build the example from

    Returns:
        dict: A dictionary of features

    """
    try:
        features = {
            'id': key,
            'label': label,
            'feat': img_bytes,
        }
        yield features

    except Exception as e:
        examples_failed.inc()
        logging.error(e, exc_info=True)
        pass


def partition_fn(example, num_partitions):
    """Deterministic partition function that partitions examples based on
    hashing.

    Args:
        example (dict): a dictionary with at least one key id

    Returns:
        int: an integer representing the partition in which the
        example is put (based on the key id)

    """
    distribution = [80, 10, 10]

    bucket = hash(str(example['id'])) % np.sum(distribution)

    if bucket < distribution[0]:
        partition_train.inc()
        return 0
    elif bucket < distribution[0] + distribution[1]:
        partition_validation.inc()
        return 1
    else:
        partition_test.inc()
        return 2


def parse_arguments(argv):
    """Parse command line arguments

    Args:
        argv (list): list of command line arguments including program name

    Returns:
        The parsed arguments as returned by argparse.ArgumentParser

    """
    parser = argparse.ArgumentParser(description='Runs Preprocessing.')

    parser.add_argument('--project_id',
                        default=PROJECT_ID,
                        help='The project to which the job will be submitted.')

    parser.add_argument('--cloud',
                        action='store_true',
                        help='Run preprocessing on the cloud.')

    parser.add_argument('--output_dir',
                        default=TFRECORD_DIR,
                        help=('Google Cloud Storage or Local directory in '
                              'which to place outputs.'))

    args, _ = parser.parse_known_args(args=argv[1:])

    return args

def get_cloud_pipeline_options(project, output_dir):
    """Get apache beam pipeline options to run with Dataflow on the cloud

    Args:
        project (str): GCP project to which job will be submitted
        output_dir (str): GCS directory to which output will be written

    Returns:
        beam.pipeline.PipelineOptions

    """
    logging.warning('Start running in the cloud')

    options = {
        'runner': 'DataflowRunner',
        'job_name': ('mlengine-flowers-{}'.format(
            datetime.now().strftime('%Y%m%d%H%M%S'))),
        'staging_location': os.path.join(output_dir, 'staging'),
        'temp_location': os.path.join(output_dir, 'tmp'),
        'project': project,
        'region': 'europe-west1',
        'zone': 'europe-west1-d',
        'autoscaling_algorithm': 'THROUGHPUT_BASED',
        'save_main_session': True,
        'setup_file': './setup.py',
    }

    return beam.pipeline.PipelineOptions(flags=[], **options)

def select_files(input_line):
    """Reads csv input lines and splits a line in uri and label

    Args:
        inputline (str): line in %s,%s format from csv file

    Returns:
       uri, label

    """
    uri = str(input_line.split(',')[0])
    label = str(input_line.split(',')[1])

    yield uri, label

def one_hot_encoding((uri, label), all_labels):
    """Transforms label into one hot encoding array

    Args:
        uri, label ((str,str))
        all_labels: array of all labels

    Returns:
       uri, labels

    """
    labels = [0]*NUM_LABELS
    labels[all_labels.index(label)] = 1
    yield uri, labels


def process_image((uri, label)):
    """Reads an image at specified uri and transforms it into pixel values
    
    Args:
        uri, label ((str,str))

    Returns:
       uri, label, image_bytes

    """
    image_bytes = read_image(uri)

    if image_bytes is not None:
        yield uri, label, image_bytes


def main(argv=None):
    """Run preprocessing as a Dataflow pipeline.

    Args:
        argv (list): list of arguments

    """
    logging.info('running main')
    args = parse_arguments(sys.argv if argv is None else argv)

    if args.cloud:
        pipeline_options = get_cloud_pipeline_options(args.project_id,
                                                      args.output_dir)
    else:
        pipeline_options = None

    pipeline = beam.Pipeline(options = pipeline_options)

    all_labels = (pipeline | 'ReadDictionary' >> beam.io.ReadFromText(
      DATA_DIR + 'dict.txt', strip_trailing_newlines=True))

    examples = (pipeline
                | 'ReadData' >> beam.io.ReadFromText(
                    'gs://cloud-ml-data/img/flower_photos/train_set.csv', strip_trailing_newlines=True)
                | 'Split' >> beam.FlatMap(select_files)
                | 'OneHotEncoding' >> beam.FlatMap(one_hot_encoding,
                                           beam.pvalue.AsIter(all_labels))
                | 'ReadImage' >> beam.FlatMap(process_image)
                | 'BuildExamples' >> beam.FlatMap(build_example))

    examples_split = examples | beam.Partition(partition_fn, 3)

    example_dict = {
        'train': examples_split[0],
        'validation': examples_split[1],
        'test': examples_split[2]
    }

    train_coder = coders.ExampleProtoCoder(schema)

    for part, examples in example_dict.items():
        examples | part + '_writeExamples' >> \
            beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=os.path.join(
                    args.output_dir, part + '_examples'),
                compression_type=beam.io.filesystem.CompressionTypes.GZIP,
                coder=train_coder,
                file_name_suffix='.tfrecord.gz')

    logging.info('running pipeline')

    pipeline.run().wait_until_finish()


if __name__ == '__main__':
    # logging.getLogger().setLevel(logging.INFO)
    main()
