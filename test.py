from khmerwordsegmentor import  KhmerWordSegmentor
t = "ចំណែកជើងទី២ នឹងត្រូវធ្វើឡើងឯប្រទេសកាតា៕"
seg = KhmerWordSegmentor()
print(seg.segment(t, model='lstm', seg_sep = '-'))
print(seg.segment(t, model='crf', seg_sep = '-'))
