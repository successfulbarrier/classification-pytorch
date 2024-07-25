# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   export.py
# @Time    :   2024/07/21 20:33:01
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   导出onnx模型

from classification import Classification

classfication = Classification()

classfication.export_onnx(onnx_path="model.onnx")
