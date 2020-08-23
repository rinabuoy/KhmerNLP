import torch 
from utils import preprocess 
from utils import postprocess
import wget
import os
from utils import seg_kcc, cleanup_str,create_kcc_features,postprocess
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import pickle

def segment(input_str,model='lstm',seg_sep = ' '):
    if model == 'lstm':
        return segment_blstm(input_str,seg_sep=seg_sep)
    elif model == 'crf':
        return segment_crf(input_str,seg_sep=seg_sep)
    else:
        return 'invalid model'

def segment_crf(input_str, model_path='sklearn_crf_model_90k-100i.sav',seg_sep = ' '):
    ts = cleanup_str(input_str)
    kccs = seg_kcc(ts)
    features = create_kcc_features(kccs)
    if not os.path.isfile(model_path):
        url = r'https://media.githubusercontent.com/media/rinabuoy/KhmerNLP/master/assets/word_segmentation_model.pt'
        wget.download(url)
    loaded_model = pickle.load(open(model_path, 'rb'))
    preds = loaded_model.predict([features])[0]
    preds = [float(p) for p in preds]
    seg = postprocess(preds,kccs,'-')
    return seg


def segment_blstm(input_str, model_path='word_segmentation_model.pt',seg_sep = ' '):
    use_gpu = torch.cuda.is_available()
    #use_gpu = False
    if(use_gpu):
        print('Inference on GPU!')
    else: 
        print('No GPU available, inference using CPU')
    if not os.path.isfile(model_path):
        url = r'https://media.githubusercontent.com/media/rinabuoy/KhmerNLP/master/assets/word_segmentation_model.pt'
        wget.download(url)
    if(use_gpu):
        model = torch.load(model_path)
    else:
        model = torch.load(model_path,map_location=torch.device('cpu'))
    model.eval()
    x,skcc = preprocess(input_str,model)
    inputs = torch.tensor(x).unsqueeze(0).long()
                
    if(use_gpu):
        inputs = inputs.cuda()
    h = model.init_hidden(1)
    val_h = tuple([each.data for each in h])
    # get the output from the model
    pred, _ = model(inputs, val_h)
    if(not use_gpu):
        pred = pred.cpu() # move to cpu

    pred = torch.sigmoid(pred)

    pred[pred<0.5] = 0.
    pred[pred>=0.5] = 1.

    return postprocess(pred,skcc,seg_sep)



