import torch

import json
import os
import numpy as np

def model_fn(model_dir):
    with open(os.path.join(model_dir, 'final.pth'), 'rb') as f:
        model = torch.jit.load(f)
    model.eval()
    return model
    
def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    request_body = json.loads(request_body)
    data = request_body['inputs']
    data = torch.tensor(data, dtype=torch.float32)
    return data

def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
    return prediction

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    cwd = os.getcwd()
    with open(cwd+'/idx_to_cls_map.json', 'r') as infile:
        labels_map = json.load(infile)
    res = np.exp(predictions.cpu().numpy().tolist())
    outcome = {'predictions': []}
    for sequential, sample in enumerate(res):
        sample_data = {}
        sample_data['sample_sequential'] = sequential
        sample_data['result'] = {'class': labels_map[str(np.argmax(sample))], 'prob': max(sample)}
        sample_data['class_probs'] = {labels_map[str(idx)]: prob for idx, prob in enumerate(sample)}

        outcome['predictions'].append(sample_data)

    return json.dumps(outcome)

