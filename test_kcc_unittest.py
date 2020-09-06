import unittest
from khmerwordsegmentor import KhmerWordSegmentor

class TestKCC(unittest.TestCase):

    def test_kcc_1(self):
        seg = KhmerWordSegmentor()
        t = "ចំណែកជើងទី២ នឹងត្រូវធ្វើឡើងឯប្រទេសកាតា៕"
        self.assertEqual(seg.segment(t, model='lstm', seg_sep = '-'), 'ចំណែក-ជើង-ទី-២-នឹង-ត្រូវ-ធ្វើឡើង-ឯ-ប្រទេស-កាតា-៕')
        self.assertEqual(seg.segment(t, model='crf', seg_sep = '-'), 'ចំណែកជើង-ទី-២-នឹង-ត្រូវ-ធ្វើឡើង-ឯ-ប្រទេស-កាតា-៕')



if __name__ == '__main__':
    unittest.main()
