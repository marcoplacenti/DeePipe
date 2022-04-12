import sagemaker
from sagemaker import get_execution_role
import json
import boto3

boto_3_session = boto3.Session()
sm_client = sess.client('sagemaker')
role = sagemaker.get_execution_role()
sm_session = sagemaker.Session(boto_session=boto3_session)

bucket_name = sm_session.default_bucket()
model_file_name = 'mymodel'
prefix = 'torchserve'

