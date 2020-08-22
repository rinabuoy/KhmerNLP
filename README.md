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


