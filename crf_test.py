from khmerwordsegmentor import segment
ts = "ចំណែកជើងទី២ នឹងត្រូវធ្វើឡើងឯប្រទេសកាតា៕"
print('lstm: ', segment(ts))
print('crf: ', segment(ts,model='crf'))



