import torch 
from utils import preprocess 
from utils import postprocess
import requests
import os
def segment(input_str, model_path='word_segmentation_model',seg_sep = ' '):
    use_gpu = torch.cuda.is_available()
    if(use_gpu):
        print('Inference on GPU!')
    else: 
        print('No GPU available, inference using CPU')
    if not os.path.isfile(model_path):
        url = 'https://drive.google.com/file/d/1tMDSuavaTxsXTUHbtxaB3AmcNIg0nZXv/view?usp=sharing'
        r = requests.get(url, allow_redirects=True)
        open('word_segmentation_model', 'wb').write(r.content)
    model = torch.load(model_path)
    model.eval()
    x,skcc = preprocess(input_str,model)
    inputs = torch.tensor(x).unsqueeze(0).long()
                
    if(use_gpu):
        inputs = inputs.cuda()
    h = model.init_hidden(1)
    val_h = tuple([each.data for each in h])
    # get the output from the model
    pred, _ = model(inputs, val_h)
    if(use_gpu):
        pred = pred.cpu() # move to cpu

    pred = torch.sigmoid(pred)

    pred[pred<0.5] = 0.
    pred[pred>=0.5] = 1.

    return postprocess(pred,skcc,seg_sep)



