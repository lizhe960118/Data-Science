#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/7/9 16:51
@Author  : LI Zhe
"""
# 把字符串变成Unicode码位的列表
symbols = '$¢£¥€¤'

# 写法1
# codes = []
# for symbol in symbols:
#     codes.append(ord(symbol))
# print(codes)

# 写法2
# codes = [ord(symbol) for symbol in symbols]
# print(codes)

beyond_ascii = [ord(s) for s in symbols if ord(s) > 127]
print(beyond_ascii)
beyond_ascii = list(filter(lambda c : c > 127, map(ord, symbols)))
print(beyond_ascii)
