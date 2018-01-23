MLEngine-Boilerplate
====================

This repository is designed to quickly get you started with new Machine Learning projects on Google Cloud Platform.
Slides: https://bit.ly/mlwithgcp

### Functionalities
- preprocessing pipeline (with Apache Beam) that runs on Cloud Dataflow or locally
- model training (with Tensorflow) that runs locally or on ML Engine
- ready to deploy saved models to deploy on ML Engine
- starter code to use the saved model on ML Engine

### Install dependencies
**Note** You will need a Linux or Mac environment with Python 2.7.x to install the dependencies [1].
Install the following dependencies:
 * Install [Cloud SDK](https://cloud.google.com/sdk/)
 * Install [gcloud](https://cloud.google.com/sdk/gcloud/)
 * ```pip install -r requirements.txt```

# Getting started

You need to complete the following parts to run the code:
- preprocess.py pipeline with your own custom preprocess steps
- model.py with your own model function according to the specification
- config.py with your project-id and databuckets
- upload data to your buckets, you can upload data/test.csv to test this code
- (optionally) task.py with more custom training steps

## Preprocess

You can run preprocess.py in the cloud using:
```
python preprocess.py --cloud
      
```

To improve efficiency you can also run the code locally on a sample of the dataset:
```
python trainer/preprocess.py
```

## Training Tensorflow model
You can submit a ML Engine training job with:
```
gcloud ml-engine jobs submit training my_job \
                --module-name trainer.task \
                --staging-bucket gs://<stating_bucket> \
                --package-path trainer
```
Testing it locally:
```
gcloud ml-engine local train --package-path trainer \
                           --module-name trainer.task
```

## Deploy your trained model
To deploy your model to ML Engine
```
gcloud ml-engine models create MODEL_NAME
gcloud ml-engine versions create VERSION --model=MODEL_NAME --origin=ORIGIN
```
To test the deployed model:
```
python predictions/predict.py
```

# ToDos
We are working to add the following functionalities:
- hypertune
- tensorflow-transform

[1] MLEngine-Boilerplate requires both Tensorflow as Apache Beam and currently Tensorflow on Windows only supports Python 3.5.x
