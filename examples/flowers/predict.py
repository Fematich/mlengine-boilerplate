from googleapiclient import discovery

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from PIL import Image
import io
import numpy as np
import sys
from trainer.util import read_image
from trainer.config import PROJECT_ID, MODEL_NAME


def get_predictions(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.
    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)


    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


if __name__ == "__main__":
    feat = read_image("gs://cloud-ml-data/img/flower_photos/dandelion/2473862606_291ae74885.jpg")

    predictions = get_predictions(
        project=PROJECT_ID,
        model=MODEL_NAME,
        instances=[
            {
                'id': "test",
                'feat': feat,
            }]
    )
    print(predictions)
