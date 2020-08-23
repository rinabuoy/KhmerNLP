from khmerwordsegmentor import segment
ts = "ចំណែកជើងទី២ នឹងត្រូវធ្វើឡើងឯប្រទេសកាតា៕"
print('Segmention by LSTM: ', segment(ts,model='lstm'))
print('Segmention by CRF: ', segment(ts,model='crf'))



