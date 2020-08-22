import unittest
from predict import segment

class TestKCC(unittest.TestCase):

    def test_kcc_1(self):
        t = "ចំណែកជើងទី២ នឹងត្រូវធ្វើឡើងឯប្រទេសកាតា៕"
        self.assertEqual(segment(t,seg_sep = '-'), 'ចំណែក-ជើង-ទី-២-នឹង-ត្រូវ-ធ្វើឡើង-ឯ-ប្រទេស-កាតា-៕', "ចំណែក-ជើង-ទី-២-នឹង-ត្រូវ-ធ្វើឡើង-ឯ-ប្រទេស-កាតា-៕")



if __name__ == '__main__':
    unittest.main()
