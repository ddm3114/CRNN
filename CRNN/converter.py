import torch
import torch.nn as nn
import collections
from collections.abc import Iterable

class strLabelConverter(object):
    # 将字符串和alphabet中的索引串相互转换
    # 输入的alphabet不需要含有空格

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1


    def encode(self, text):
        # 输入： 字符串 / 字符串list
        # 输出:
        # torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: 索引串拼接成的tensor
        # orch.IntTensor [n]: 每个索引串的长度
        
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        else:
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))


    def decode(self, t, length, raw=False):
        # 输入： （encode的输出格式）索引串的拼接tensor t、每个索引串的长度lenth
        #         raw的 假/真 表示 是/否 对字符串做压缩操作（a-ppp--l-ee  -->  apple）
        # 输出： 字符串（list）

        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts