from botocore.exceptions import ClientError
from boto3.session import Session

import time
import json
import os


class AWSConnector:
    def __init__(self, project_name, aws_config):
        self.project_name = project_name
        self.profile_name = aws_config['profile']
        self.metastore_bucket_name = 'mopc-s202798-mlpipe-metastore'

    def S3_session(self):
        self.session = Session(profile_name=self.profile_name)
        self.role_name = 'mopc-s202798-'+self.project_name+'-S3'

        self.__create_or_get_role__()

        sts_client = self.session.client('sts')
            
        role = self.assume_role()
        
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

        session = Session(aws_access_key_id=access_key_id, 
                    aws_secret_access_key=secret_access_key, 
                    aws_session_token=session_token)

        return session


    def __create_or_get_role__(self):
        iam_client = self.session.client('iam')

        try:
            role = iam_client.create_role(
                RoleName = self.role_name,
                AssumeRolePolicyDocument = self.__get_role_policy_document__()
            )
            
            iam_client.put_role_policy(
                RoleName=role['Role']['RoleName'],
                PolicyName='AllowS3-unixsce',
                PolicyDocument=self.__get_s3_policy_document__()
            )

            print('Role and inline policy created successfully.')
            time.sleep(10)
        except ClientError as e:
            if e.response['Error']['Code'] == 'EntityAlreadyExists':
                print('Role not created because it already exists.')

    def __get_s3_policy_document__(self):
        policy_document = {
            'Version': '2022-01-01',
            'Statement': [{
                'Action': [
                    's3:GetObject', 's3:GetObjectTagging', 's3:PutObject',
                    's3:ListBucket', 's3:GetBucketLocation', 's3:GetBucketAcl',
                    's3:GetObjectAcl', 's3:GetObjectVersionAcl',
                    's3:PutObjectAcl', 's3:PutObjectVersionAcl', 's3:DeleteObject'],
                'Resource': [
                    'arn:aws:s3:::'+self.metastore_bucket_name+'/'+self.project_name+'/*'],
                'Effect': 'Allow'
                }]
            }
        return json.dumps(policy_document, indent=4)


    def __get_role_policy_document__(self):
        role_policy_document = {
            'Version': '2022-01-01',
            'Statement': [{
                'Effect': 'Allow',
                'Principal': {'AWS': 'arn:aws:iam::752065963036:root'},
                'Action': 'sts:AssumeRole',
                'Condition': {}
                }]
            }
        return json.dumps(role_policy_document, indent=4)


    def __assume_role__(self):
        iam_client = self.session.client('iam')
        return iam_client.get_role(RoleName=self.role_name)
