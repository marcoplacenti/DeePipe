from src.pipe.AWSConnector import AWSConnector
from src.pipe.DeePipe import DeePipe

import argparse

def consume_model():
    import boto3
    import json
    import os
    import numpy as np
    import cv2
    from torchvision import transforms
    import torch
    
    endpoint = 'MOPCTestEndpoint'
    connector = AWSConnector(project_name="MyProject")
    session, _ = connector.get_sagemaker_role()
    runtime = session.client('sagemaker-runtime')

    images = [
        './data/source/MNISTMini/2/35.jpg',
        './data/source/MNISTMini/4/6.jpg',
        './data/source/MNISTMini/7/17.jpg'
    ]
    for path in images:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if len(image.shape) == 2:
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        image = np.transpose(image, axes=(2, 0, 1))
        image = torch.from_numpy(image)/255
        transform = transforms.Resize((28, 28))
        image = transform(image)
        image = np.reshape(image, (1, 1, 28, 28))
        dummy_data = {"inputs": image.tolist()}
        # Send image via InvokeEndpoint API
        response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/json', Body=json.dumps(dummy_data))

        # Unpack response
        result = json.loads(response['Body'].read().decode())
        print(result)
    


def mode_1(config_dict):
    ### OPTION 1 ###
    print("Option 1")
    DeePipe(config_dict)


def mode_2():
    ### OPTION 2 ###
    print("Option 2")
    pipe = DeePipe(name='ImageClassificationV2', task='classification')
    pipe.preproc_data(location='data/source/MNISTMini/', img_res=[28,28], greyscale=False, test_size=0.2, folds=2)
    pipe.train(max_epochs=1, batch_size=[64, 128], optimizer='Adam', learning_rate=[0.0001, 0.01], number_trials=3)
    pipe.eval()
    pipe.deploy()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML Pipe NN.')
    parser.add_argument("--config", help="Provide path to configuration file")
    args = parser.parse_args()

    #mode_1(args.config)
    #mode_2()

    consume_model()
    
