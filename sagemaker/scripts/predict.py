import argparse
import os
import sys
from io import StringIO
from six import BytesIO
from model import SimpleNet

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


# Accepts and returns numpy data
CONTENT_TYPE = 'application/x-npy'


def model_fn(model_dir):
    """Required function for SageMaker to run inference.

    Before a model can be served, it must be loaded. The SageMaker PyTorch model server loads your
    model by invoking a model_fn function that you must provide in your script when you are not
    using Elastic Inference.
    """
    print("loading model from {}".format(model_dir))
    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(model_info['input_dim'], 
                      model_info['hidden_dim'], 
                      model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    return model.to(device)


def input_fn(serialized_input_data, content_type):
    """Required function for SageMaker to run inteference.

    It de-serializes request bytes into numpy array.  
    """
    if content_type == CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


def output_fn(prediction_output, accept):
    """Required function for SageMaker to run inteference.

    It serializes model output into bytes.
    """
    if accept == CONTENT_TYPE:
        buffer = BytesIO()
        np.save(buffer, prediction_output)
        return buffer.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model):
    """Required function for SageMaker to run inteference.

    It is invoked to perform predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process input_data so that it is ready to be sent to our model
    # convert data to numpy array then to Tensor
    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    # Put model into evaluation mode
    model.eval()

    # Forward propagate the input and produce an output
    out = model(data)

    # The variable `result` should be a numpy array; a single value 0-1
    result = out.cpu().detach().numpy()

    return result
