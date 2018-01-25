from googleapiclient import discovery


def get_predictions(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): GCP project where the ML Engine Model is deployed.
        model (str): model name
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version (str) version of the model to target

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
    # TODO change get_predictions to test your model
    # use read_image for uri: "gs://cloud-ml-data/img/flower_photos/dandelion/2473862606_291ae74885.jpg"
    predictions = get_predictions(
        project="project-id",
        model="mlengine_boilerplate",
        instances=[
            {
                'id': "a12",
                'feat': [138.0, 30.0, 66.0],
            }]
    )
    print(predictions)
