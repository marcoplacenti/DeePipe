import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import pytorch_lightning as pl

import json
import os

def model_fn(model_dir):
    print("LOADING MODEL")
    print(model_dir+'/final.pth')
    print(os.listdir(model_dir))
    with open(os.path.join(model_dir, 'final.pth'), 'rb') as f:
        model = torch.jit.load(f)
    model.eval()
    return model
    
def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    data = request_body['inputs']
    data = torch.tensor(data, dtype=torch.float32)
    return data

def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
    return prediction

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)


model = model_fn('./src/pipe/inference/')
import numpy as np
import cv2
from torchvision import transforms

image = cv2.imread('./data/source/MNISTMini/2/35.jpg', cv2.IMREAD_GRAYSCALE)
if len(image.shape) == 2:
    image = np.reshape(image, (image.shape[0], image.shape[1], 1))
image = np.transpose(image, axes=(2, 0, 1))
image = torch.from_numpy(image)/255
transform = transforms.Resize((28, 28))
image = transform(image)
image = np.reshape(image, (1, 1, 28, 28))
dummy_data = {"inputs": image.tolist()}
#dummy_data = {"inputs": np.random.rand(16, 1, 28, 28).tolist()}
input_object = input_fn(dummy_data, 'application/json')
preds = predict_fn(input_object, model)
print(np.argmax(np.exp(preds)))
