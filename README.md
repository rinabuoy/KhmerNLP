[![DOI](https://zenodo.org/badge/289483297.svg)](https://zenodo.org/badge/latestdoi/289483297)

![KCC-Level Network](https://github.com/rinabuoy/KhmerNLP/blob/master/assets/KCCNetwork.PNG)

# Khmer Word Segmentation
Word segmentation on Khmer texts is a challenging task since in Khmer texts, there is
no explicit word delimiters such as a space.
This is not the case for Latin languages such
French or English. In Khmer texts, characters are written from left to right consecutively with optional space between words.
Another challenge is that most words can be
co-located to form a new word. For example, សម្តេច (your highness) can be a proper
word by itself or can be split into two words:
ស(white)and ម្តេច (how) both of which are
proper words by themselves [1; 2; 3].

# Khmer Character Cluster(KCC)

The concept of Khmer Character Cluster
(KCC) was introduced in [3; 6]. KCC is
the inseparable sequence of characters. In
Khmer writing system, a vowel cannot be by
itself; a vowel must be placed after a consonant. Here are a few examples showing that
a word is a combination of KCCs [6]:
* សាលាក្តី has 3 KCCs (សា+លា+ក្តី)
* ចុកចាប់ has 4 KCCs (ចុ+ក+ចា+ប់)
* ស្ត្រី has 1 KCC only.

![KCC Rules](https://github.com/rinabuoy/KhmerNLP/blob/master/assets/KCCrule.PNG)


# BiLSTM Networks for Khmer Word Segmentation

## Character Level vs KCC Level

### Character Level

![Character-Level Network](https://github.com/rinabuoy/KhmerNLP/blob/master/assets/CharacterLevelNetwork.PNG)


### KCC Level

![KCC-Level Network](https://github.com/rinabuoy/KhmerNLP/blob/master/assets/KCCNetwork.PNG)

# Running Word Segmentation Usign Pre-Trained KCC Network


```python

from khmerwordsegmentor import  segment
from khmerwordsegmentor import segment
ts = "ចំណែកជើងទី២ នឹងត្រូវធ្វើឡើងឯប្រទេសកាតា៕"
print('Segmention by LSTM: ', segment(ts,model='lstm'))
print('Segmention by CRF: ', segment(ts,model='crf'))


#Inference on GPU!
#ចំណែក-ជើង-ទី-២-នឹង-ត្រូវ-ធ្វើឡើង-ឯ-ប្រទេស-កាតា-៕

```
# TODO

* Character Level Model 
* Conditional Random Field Model


# Citation
```
@misc{RinaB2020,
  author = {Rina Buoy and Sokchea Kor},
  title = {Khmer Word Segmentation Using BiLSTM Networks},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rinabuoy/KhmerNLP}}
}
```

# References

[1] Vichet Chea, Ye Kyaw Thu, Chenchen
Ding, Masao Utiyama, Andrew Finch,
and Eiichiro Sumita. Khmer word
segmentation using conditional random
fields. Khmer Natural Language Processing, 2015.

[2] Narin Bi and Nguonly Taing. Khmer
word segmentation based on bidirectional maximal matching for
plaintextand microsoft word. Signal and
Information Processing Association Annual Summit and Conference (APSIPA),
2014.

[3] Chea Sok Huor, Top Rithy, Ros Pich
Hemy, Vann Navy, Chin Chanthirith,
and Chhoeun Tola. Word bigram vs
orthographic syllable bigram in khmer
word. PAN Localization Team, 2007.

[4] Ye Kyaw Thu, Vichet Chea, Andrew
Finch, Masao Utiyama, and Eiichiro
Sumita. A large-scale study of statistical
machine translation methods for Khmer
language. In Proceedings of the 29th Pacific Asia Conference on Language, Information and Computation, pages 259–269,
Shanghai, China, October 2015.

[5] Phylypo Tum. Word Segmentation of
Khmer Text Using Conditional Random
Fields, June 2020.

[6] Chea Sok Huor and Top Rithy. Detection and correction of homophonous error
word for khmer language. 2007.

[7] Andrej Karpathy, Justin Johnson, and
Li Fei-Fei. Visualizing and understanding recurrent networks, 2015.

[8] Zhiheng Huang, Wei Xu, and Kai Yu.
Bidirectional lstm-crf models for sequence tagging, 2015.
