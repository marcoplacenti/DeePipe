from botocore.exceptions import ClientError
import sagemaker
import boto3
from sagemaker import get_execution_role, Session
from sagemaker.pytorch import PyTorchModel

import time
import json
import os

import wandb

import tarfile


class AWSConnector:
    def __init__(self, project_name):
        self.project_name = project_name
        self.metastore_bucket_name = 'mopc-s202798-mlpipe-metastore'
        self.datasets_bucket_name = 'mopc-s202798-mlpipe-datasets'
        self.sagemaker_bucket_name = 'mopc-s202798-mlpipe-sagemaker'

    def get_contrib_session(self):
        return boto3.session.Session(
                aws_access_key_id=os.environ['CLOUD_DEVOPS_USER_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['CLOUD_DEVOPS_USER_SECRET_ACCESS_KEY'],
                region_name='eu-central-1')

    def S3_session(self):
        self.contrib_session = self.get_contrib_session()
        role_name = 'mopc-s202798-'+self.project_name+'-S3'

        iam_client = self.contrib_session.client('iam')
        self.__create_or_get_role__(iam_client, role_name)
        self.put_role_policy(iam_client, role_name, 
                            'AllowS3-MLPipe', 
                            self.__get_s3_policy_document__())

        sts_client = self.contrib_session.client('sts')
            
        role = self.__assume_role__(role_name)
        
        response = sts_client.assume_role(
            DurationSeconds=3600,
            RoleArn=role['Role']['Arn'],
            RoleSessionName='uploadS3Objects'
        )

        access_key_id = response['Credentials']['AccessKeyId']
        secret_access_key = response['Credentials']['SecretAccessKey']
        session_token = response['Credentials']['SessionToken']

        os.environ['AWS_ACCESS_KEY_ID'] = access_key_id
        os.environ['AWS_SECRET_ACCESS_KEY'] = secret_access_key
        os.environ['AWS_SESSION_TOKEN'] = session_token

        role_session = boto3.session.Session(aws_access_key_id=access_key_id, 
                    aws_secret_access_key=secret_access_key, 
                    aws_session_token=session_token,
                    region_name='eu-central-1')

        del self.contrib_session
        return role_session

    def get_sagemaker_role(self):
        self.contrib_session = self.get_contrib_session()

        role_name = 'mopc-s202798-'+self.project_name+'-Sagemaker'
        #role_name = 'AmazonSageMaker-ExecutionRole-20210119T155723'
        #role_name = 'sagemaker-execution-role'

        iam_client = self.contrib_session.client('iam')
        self.__create_or_get_role__(iam_client, role_name)
        response = iam_client.attach_role_policy(
            RoleName=role_name, PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess')

        self.put_role_policy(iam_client, role_name, 
                            'AllowSagemaker-MLPipe', 
                            self.__get_sagemaker_policy_document__())

        sts_client = self.contrib_session.client('sts')

        role = self.__assume_role__(role_name)
        
        response = sts_client.assume_role(
            DurationSeconds=3600,
            RoleArn=role['Role']['Arn'],
            RoleSessionName='uploadS3Objects'
        )

        access_key_id = response['Credentials']['AccessKeyId']
        secret_access_key = response['Credentials']['SecretAccessKey']
        session_token = response['Credentials']['SessionToken']

        role_session = boto3.session.Session(
                            aws_access_key_id=access_key_id, 
                            aws_secret_access_key=secret_access_key, 
                            aws_session_token=session_token,
                            region_name='eu-central-1')

        del self.contrib_session
        return role_session, role


    def __create_or_get_role__(self, iam_client, role_name):
        try:
            role = iam_client.create_role(
                RoleName = role_name,
                AssumeRolePolicyDocument = self.__get_role_policy_document__()
            )

        except Exception as e:
            if e.response['Error']['Code'] == 'EntityAlreadyExists':
                pass
            else:
                print(e)
                exit()

    def put_role_policy(self, iam_client, role_name, policy_name, policy_document):
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=policy_document
        )
        time.sleep(10)

    def __get_s3_policy_document__(self):
        policy_document = {
            'Version': '2012-10-17',
            'Statement': [
                {
                    'Action': [
                        's3:GetObject', 's3:PutObject'],
                    'Resource': [
                        'arn:aws:s3:::'+self.metastore_bucket_name+'/*',
                        'arn:aws:s3:::'+self.datasets_bucket_name+'/*'],
                    'Effect': 'Allow'
                },
                {
                    'Action': [
                        's3:ListBucket'],
                    'Resource': [
                        'arn:aws:s3:::'+self.metastore_bucket_name,
                        'arn:aws:s3:::'+self.datasets_bucket_name],
                    'Effect': 'Allow'
                }]
            }

        return json.dumps(policy_document, indent=4)

    def __get_sagemaker_policy_document__(self):
        policy_document = {
            'Version': '2012-10-17',
            'Statement': [
                {
                    'Action': [
                        'sagemaker:*', 's3:*',
                        'sagemaker:PutObject', 'sagemaker:PutObjectAcl',
                        'sagemaker:GetObject', 'sagemaker:GetObjectAcl',
                        'sagemaker:DeleteObject', 'sagemaker:ListBucket',
                        'sagemaker:HeadObject', 's3:HeadObject'],
                    'Resource': [
                        'arn:aws:s3:::'+self.sagemaker_bucket_name+'/*',
                        'arn:aws:s3:::'+self.sagemaker_bucket_name+'/',
                        'arn:aws:s3:::'+self.sagemaker_bucket_name],
                    'Effect': 'Allow'
                }]
            }

        return json.dumps(policy_document, indent=4)


    def __get_role_policy_document__(self):
        role_policy_document = {
            'Version': '2012-10-17',
            'Statement': [{
                'Effect': 'Allow',
                'Principal': {
                    'AWS': 'arn:aws:iam::752065963036:root',
                    'Service': 'sagemaker.amazonaws.com'},
                'Action': 'sts:AssumeRole',
                'Condition': {}
                }]
            }
        return json.dumps(role_policy_document, indent=4)


    def __assume_role__(self, role_name):
        iam_client = self.contrib_session.client('iam')
        return iam_client.get_role(RoleName=role_name)


    def upload_artifacts(self, artifact_type, path, project):
        session = self.S3_session()
        s3_client = session.client('s3')

        run = wandb.init(project=project['name'], name=project['experiment'], entity='dma')

        for root, _, files in os.walk(path):
            for obj in files:
                obj_name = project['name']+'/'+artifact_type+'/'
                if artifact_type == 'data':
                    obj_name = obj_name + obj
                elif artifact_type == 'trials':
                    obj_name = obj_name + '/'.join(root.split('/')[2:])+'/'+obj
                elif artifact_type == 'models':
                    obj_name = obj_name + obj
                else:
                    raise ValueError

                s3_client.upload_file(os.path.join(root, obj), 
                                    self.metastore_bucket_name, 
                                    obj_name)

                artifact = wandb.Artifact(obj, type=artifact_type)
                artifact.add_reference('s3://'+self.metastore_bucket_name+'/'+obj_name)
                run.log_artifact(artifact)

        del s3_client
        del session
        #wandb.finish()

    def download_data(self, bucket_uri, destination):
        session = self.S3_session()
        s3_client = session.client('s3')

        print("Downloading data, this may take a while...")
        self.download_dir('MNISTMini/', 'data/', datasets_bucket_name, s3_client)

    def download_dir(self, prefix, local, bucket, client):
        """
        params:
        - prefix: pattern to match in s3
        - local: local path to folder in which to place files
        - bucket: s3 bucket with target contents
        - client: initialized s3 client object
        """
        keys = []
        dirs = []
        next_token = ''
        base_kwargs = {
            'Bucket':bucket,
            'Prefix':prefix,
        }
        while next_token is not None:
            kwargs = base_kwargs.copy()
            if next_token != '':
                kwargs.update({'ContinuationToken': next_token})
            results = client.list_objects_v2(**kwargs)
            contents = results.get('Contents')
            for i in contents:
                k = i.get('Key')
                if k[-1] != '/':
                    keys.append(k)
                else:
                    dirs.append(k)
            next_token = results.get('NextContinuationToken')
        for d in dirs:
            dest_pathname = os.path.join(local, d)
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
        for k in keys:
            dest_pathname = os.path.join(local, k)
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            client.download_file(bucket, k, dest_pathname)

    def deploy(self):
        with tarfile.open('./tmp/models/myModel.tar.gz', "w:gz") as tar:
            #tar.add('./tmp/models/final.pth', arcname='./final.pth')
            tar.add('./src/pipe/inference/', arcname='.')
            tar.add('./src/models/', arcname='./src/models/')

        session, role = self.get_sagemaker_role()
        
        #self.contrib_session = self.get_contrib_session()
        #role = self.__assume_role__(role)
        #role = role['Role']['Arn']
        print("Role: ", role['Role']['Arn'])

        
        sess = Session(boto_session=session, default_bucket=self.sagemaker_bucket_name)

        model_data = sess.upload_data('./tmp/models/myModel.tar.gz', 
                                bucket=self.sagemaker_bucket_name,
                                key_prefix='model/pytorch')

        print("tar.gz file: ", model_data)

        from sagemaker.pytorch.model import PyTorchPredictor
        model = PyTorchModel(
            entry_point='inference.py',
            #source_dir='./src/pipe/inference/code/',
            role=role['Role']['Arn'],
            sagemaker_session=sess,
            model_data=model_data,
            framework_version="1.5.0",
            py_version='py3',
            predictor_cls=PyTorchPredictor,
        )

        from sagemaker.serializers import JSONSerializer
        from sagemaker.deserializers import JSONDeserializer
        predictor = model.deploy(
            instance_type='ml.m4.xlarge',
            initial_instance_count=1,
            #endpoint_name='mopc-ImageClassification',
            #accelerator_type='ml.eia2.medium'
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )

        print("Endpoint name: ", predictor.endpoint_name)

        import random
        import numpy as np

        dummy_data = {"inputs": np.random.rand(16, 1, 28, 28).tolist()}
        
        res = predictor.predict(dummy_data)
        print("Predictions:", res)
