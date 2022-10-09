import utils.config as config
import subprocess

from typing import NamedTuple
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component, 
                        OutputPath, 
                        InputPath)

from kfp.v2 import compiler
from google.cloud import bigquery
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from google_cloud_pipeline_components import aiplatform as gcc_aip

def cli(query, capture_output=False, text=False):
    result = subprocess.run([query], shell=True, capture_output=capture_output, text=text)
    return result

cli(f"gcloud config set project {config.project}")

cli("gcloud services enable compute.googleapis.com              \
                            containerregistry.googleapis.com    \
                            aiplatform.googleapis.com           \
                            cloudbuild.googleapis.com           \
                            cloudfunctions.googleapis.com       \
                            dataflow.googleapis.com")

# Set bucket name
BUCKET_NAME="gs://crown-of-thorns-data"

# Create bucket
# PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root_wine/"

#!gcloud auth login if needed