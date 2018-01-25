#!/usr/bin/python
import argparse
import logging
import os
import sys
from datetime import datetime

import apache_beam as beam
from apache_beam.io import filesystem
from apache_beam.io import tfrecordio
from apache_beam.metrics import Metrics

from tensorflow_transform import coders

from trainer.config import BUCKET, DATA_DIR, PROJECT_ID, TFRECORD_DIR
from trainer.util import schema

partition_train = Metrics.counter('partition', 'train')
partition_validation = Metrics.counter('partition', 'validation')
partition_test = Metrics.counter('partition', 'test')
examples_failed = Metrics.counter('build', 'failed')


def build_example(raw_in):
    """Build a dictionary that contains all the features and label to store
    as TFRecord

    Args:
        raw_in: raw data to build the example from

    Returns:
        dict: A dictionary of features

    """
    try:
        elements = raw_in.split(',')
        key = elements[0]
        label = float(elements[1])
        feat = [float(el) for el in elements[2:]]
        features = {
            'id': key,
            'label': label,
            'feat': feat,
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
        num_partitions: number of partitions, unused but enforced parameter

    Returns:
        int: an integer representing the partition in which the
        example is put (based on the key id)

    """
    distribution = [80, 10, 10]

    bucket = hash(str(example['id'])) % sum(distribution)

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
                        default=BUCKET,
                        help=('Google Cloud Storage or Local directory in '
                              'which to place outputs.'))

    args, _ = parser.parse_known_args(args=argv[1:])

    return args

def split_line(input_line):
    """Reads csv input lines and splits a line in uri and label

    Args:
        inputline (str): line in %s,%s format from csv file

    Returns:
       uri, label

    """
    #TODO implement function 
    
def one_hot_encoding((uri, label), all_labels):
    """Transforms label into one hot encoding array

    Args:
        uri, label ((str,str))
        all_labels: array of all labels

    Returns:
       uri, labels

    """
    #TODO implement function 

def process_image((uri, label)):
    """Reads an image at specified uri and transforms it into pixel values
    uses read_image from util

    Args:
        uri, label ((str,str))

    Returns:
       uri, label, pixel_values

    """
    #TODO implement function (uses read_image from util)


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
        'job_name': ('mlengine-boilerplate-{}'.format(
            datetime.now().strftime('%Y%m%d%H%M%S'))),
        'staging_location': os.path.join(BUCKET, 'staging'),
        'temp_location': os.path.join(BUCKET, 'tmp'),
        'project': project,
        'region': 'europe-west1',
        'zone': 'europe-west1-d',
        'autoscaling_algorithm': 'THROUGHPUT_BASED',
        'save_main_session': True,
        'setup_file': './setup.py',
    }

    return beam.pipeline.PipelineOptions(flags=[], **options)


def main(argv=None):
    """Run preprocessing as a Dataflow pipeline.

    Args:
        argv (list): list of arguments

    """
    args = parse_arguments(sys.argv if argv is None else argv)

    if args.cloud:
        pipeline_options = get_cloud_pipeline_options(args.project_id,
                                                      args.output_dir)
    else:
        pipeline_options = None

    pipeline = beam.Pipeline(options=pipeline_options)

    all_labels = (pipeline | 'ReadDictionary' >> beam.io.ReadFromText(
        DATA_DIR + 'dict.txt', strip_trailing_newlines=True))

     # TODO: adapt pipeline to use new functions defined above
     # use all_labels (array of all possible labels as sideinput in the pipeline)
    examples = (pipeline
                # | 'ReadData' >> beam.Create(open('data/test.csv')
                #                             .readlines()[1:])
                | 'ReadData' >> beam.io.ReadFromText(DATA_DIR + '*',
                                                     skip_header_lines=1)
                | 'BuildExamples' >> beam.FlatMap(build_example))

    examples_split = examples | beam.Partition(partition_fn, 3)

    example_dict = {
        'train': examples_split[0],
        'validation': examples_split[1],
        'test': examples_split[2]
    }

    for part, examples in example_dict.items():
        examples | part + '_writeExamples' >> tfrecordio.WriteToTFRecord(
            file_path_prefix=os.path.join(TFRECORD_DIR, part + '_examples'),
            compression_type=filesystem.CompressionTypes.GZIP,
            coder=coders.ExampleProtoCoder(schema),
            file_name_suffix='.tfrecord.gz')

    pipeline.run().wait_until_finish()


if __name__ == '__main__':
    main()
