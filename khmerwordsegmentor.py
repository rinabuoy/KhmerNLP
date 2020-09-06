import sys
import os
import torch
from utils import preprocess
from utils import postprocess
import wget
import os
from utils import seg_kcc, cleanup_str,create_kcc_features,postprocess
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import pickle

class KhmerWordSegmentor(object):
    """docstring for KhmerWordSegmentor"""
    def __init__(self, crf_model_path = None, bilstm_model_path = None):
        super(KhmerWordSegmentor, self).__init__()

        if crf_model_path is None:
          crf_model_path = os.path.join(os.path.dirname(__file__), 'assets/sklearn_crf_model_90k-100i.sav')
        file = open(crf_model_path, 'rb')
        self.crfModel = pickle.load(file)
        file.close()

        self.use_gpu = torch.cuda.is_available()
        #use_gpu = False
        if(self.use_gpu):
            print('Inference on GPU!')
        else:
            print('No GPU available, inference using CPU')

        if bilstm_model_path is None:
          bilstm_model_path = os.path.join(os.path.dirname(__file__), 'assets/word_segmentation_model.pt')

        if(self.use_gpu):
            self.bilstmModel = torch.load(bilstm_model_path)
        else:
            self.bilstmModel = torch.load(bilstm_model_path, map_location=torch.device('cpu'))
        self.bilstmModel.eval()


    def segment(self, input_str, model='lstm', seg_sep = ' '):
        if model == 'lstm':
            return self.segment_blstm(input_str,seg_sep=seg_sep)
        elif model == 'crf':
            return self.segment_crf(input_str,seg_sep=seg_sep)
        else:
            return 'invalid model'

    def segment_crf(self, input_str, seg_sep = ' '):
        ts = cleanup_str(input_str)
        kccs = seg_kcc(ts)
        features = create_kcc_features(kccs)
        preds = self.crfModel.predict([features])[0]
        preds = [float(p) for p in preds]
        seg = postprocess(preds,kccs, seg_sep)
        return seg

    def segment_blstm(self, input_str, seg_sep = ' '):
        x,skcc = preprocess(input_str,self.bilstmModel)
        inputs = torch.tensor(x).unsqueeze(0).long()

        if(self.use_gpu):
            inputs = inputs.cuda()
        h = self.bilstmModel.init_hidden(1)
        val_h = tuple([each.data for each in h])
        # get the output from the model
        pred, _ = self.bilstmModel(inputs, val_h)
        if(not self.use_gpu):
            pred = pred.cpu() # move to cpu

        pred = torch.sigmoid(pred)

        pred[pred<0.5] = 0.
        pred[pred>=0.5] = 1.

        return postprocess(pred,skcc,seg_sep)



