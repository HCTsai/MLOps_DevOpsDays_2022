'''
Created on 2022年6月21日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
import unittest
from features import feature_drift_detector

class TestFeatureDrift(unittest.TestCase):
    #簡體轉換成繁體       
    def test_drift(self):
        test_list = ["a這依據是非 化沒有特別ˋ意義'","今日我們需發展專業","面對外在的競爭","謝謝","aaaaaaaaaaaaa"]
        res = feature_drift_detector.detect_feature_drift(test_list)
        # "謝謝","aaaaaaaaaaaaa"
        self.assertTrue(2 == len(res))
   
if __name__ == '__main__':
    unittest.main()