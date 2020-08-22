[![DOI](https://zenodo.org/badge/289483297.svg)](https://zenodo.org/badge/latestdoi/289483297)

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

The pytorch pre-trained model can be obtained from  https://drive.google.com/file/d/1tMDSuavaTxsXTUHbtxaB3AmcNIg0nZXv/view?usp=sharing. 

```python

from predict import  segment
t = "ចំណែកជើងទី២ នឹងត្រូវធ្វើឡើងឯប្រទេសកាតា៕"
print(segment(t,seg_sep = '-'))

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

