'''
Created on 2021年4月21日

@author: hsiaochien.tsai
'''
from texttool.chinese_transform_dict import zh2han, han2zh, cn2zh, sg2zh
from collections import defaultdict
class ChineseTransform:
    def __init__(self):
        self.algorithm = "han2zh"
        # dict转为defaultdict
        # 繁體轉簡體
        self.han2zhs = self.load_han_zh_dict([han2zh, cn2zh, sg2zh])
        # 簡體轉繁體
        inv_cn2zh = {v: k for k, v in cn2zh.items()}
        self.zh2hans = self.load_han_zh_dict([zh2han,inv_cn2zh])

    def load_han_zh_dict(self, dicts):
        """
         多個 dictionary 轉成一個dictionary
        :param dicts: list<dict>, like [{"丟": "丢"}, {"並": "并"}]
        :return: dict, like {"丟": "丢", "並": "并"}
        """
        dict_han_zh = defaultdict()
        for ds in dicts:
            for k, v in ds.items():
                dict_han_zh[k] = v
        return dict_han_zh

    def trad2simp(self, text, len_max=11):
        """
        繁体字转简体字, 反向最大切词
        :param sentence: str, like '雪鐵龍'
        :param len_max: int, like 9
        :return: str, like '雪铁龙'
        """
        len_sen = len(text)
        i = len_sen
        res = [""]
        while i > 0:  # while判断条件
            flag = False  # flag标志位,确定有没有在字典里边的单字词或多字词
            for j in range(max(0, i - len_max), i):  # 遍历从句子末尾向前可能成词的部分, 从最后i-len_max算起
                word_maybe = text[j:i]  # 正向可能成词的语
                if word_maybe in self.han2zhs:  # 是否在字典里边
                    i = j  # 成词前标志i向后移动
                    flag = True  # flag标志位变化
                    res.append(self.han2zhs.get(word_maybe))
                    break  # 成词则跳出循环
            if not flag:  # 未选中后单个字的情况
                i -= 1
                res_i = self.han2zhs.get(text[i])
                if res_i:
                    res.append(res_i)
                else:
                    res.append(text[i])
        res.reverse()
        return "".join(res)
    def simp2trad(self, text, len_max=5):
        """
        简体字转繁体字, 反向最大切词
        :param sentence: str, like '大漠帝国'
        :param len_max: int, like 32
        :return: yield
        """
        len_sen = len(text)
        i = len_sen
        res = [""]
        while i > 0:  # while判断条件
            flag = False  # flag标志位,确定有没有在字典里边的单字词或多字词
            for j in range(max(0, i - len_max), i):  # 遍历从句子末尾向前可能成词的部分, 从最后i-len_max算起
                word_maybe = text[j:i]  # 正向可能成词的语
                if word_maybe in self.zh2hans:  # 是否在字典里边
                    i = j  # 成词前标志i向后移动
                    flag = True  # flag标志位变化
                    res.append(self.zh2hans.get(word_maybe))
                    break  # 成词则跳出循环
            if not flag:  # 未选中后单个字的情况
                i -= 1
                res_i = self.zh2hans.get(text[i])
                if res_i:
                    res.append(res_i)
                else:
                    res.append(text[i])
        res.reverse()
        return "".join(res)
if __name__ == '__main__':
    trans = ChineseTransform()
    text = " 請給我滑鼠與螢幕"
    res_han2zh = trans.trad2simp(text)
    text = " 请给我光盘 aaa 光碟 bbb 謝謝"
    res_zh2han = trans.simp2trad(text)
    print(res_han2zh)
    print(res_zh2han)
    


