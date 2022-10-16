import pandas as pd
import numpy as np
"""
@author: 徐小东&郁晴
"""
def chi2(arr):
    '''
    计算卡方值
    arr：频数统计表，二维numpy数组
    '''
    assert (arr.ndim == 2)
    # assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达示为假。
    # 可以理解assert断言语句为raise-if-not，用来测试表示式，其返回值为假，就会触发异常。

    R_N = arr.sum(axis=1)  # 计算每行总频数
    C_N = arr.sum(axis=0)  # 计算每列总频数
    N = arr.sum()  # 总频数

    E = np.ones(arr.shape) * C_N / N  # 计算期望频数
    E = (E.T * R_N).T
    square = (arr - E) ** 2 / E
    square[E == 0] = 0  # 当期望频数为0时，作为分母无意义，不计入卡方值

    v = square.sum()  # 卡方值
    return v


def chiMerge(df, col, target, max_groups, threshold=None):
    '''
    卡方分箱
    df: pandas DataFrame 数据集
    col: 需要分箱的变量名（数值型）
    target：类标签
    max_groups: 最大分组数
    threshold： 卡方阈值，如果未指定max_groups，默认使用置信度95%设置threshold
    return：包括各组的起始值的列表
    '''
    freq_tab = pd.crosstab(df[col], df[target])

    # 转成numpy数组用于计算
    freq = freq_tab.values

    # 初始分组切分点，每个变量值都是切分点，每组中只包含一个变量值
    # 分组区间是左闭右开的，如cutoffs = [1,2,3]，表示区间，[1,2),[2,3).[3,3+)
    cutoffs = freq_tab.index.values

    # 如果没有指定最大分组
    if max_groups is None:
        # 如果没有指定卡方阈值，就以95%的置信度（自由度为类数目-1）设定阈值
        if threshold is None:
            # 类数目
            cls_num = freq.shape[-1]
            threshold = chi2.isf(0.05, df=cls_num - 1)

    while True:
        minvalue = None
        minidx = None
        # 从第一组开始，依次计算两组卡方值，并判断是否小于当前最小的卡方
        for i in range(len(freq) - 1):
            v = chi2(freq[i:i + 2])
            if minvalue is None or minvalue > v:  # 小于当前最小卡方，更换最小值
                minvalue = v
                minidx = i

        # 如果最小卡方值小于阈值，则合并最小卡方值的相邻两组，并继续循环
        if (max_groups is not None and max_groups < len(freq)) or (threshold is not None and minvalue < threshold):
            # minidx 后一行合并到minidx
            tmp = freq[minidx] + freq[minidx + 1]
            freq[minidx] = tmp
            # 删除minidx后一行
            freq = np.delete(freq, minidx + 1, 0)
            # 删除对应的切分点
            cutoffs = np.delete(cutoffs, minidx + 1, 0)

        else:  # 最小卡方值不小于阈值，停止合并
            break
    return cutoffs


def value2group(x, cutoffs):
    '''
    将变量的值转换成相应的组
    x: 需要转换到分组的值
    cutoffs：各组的起始值
    return： x对应的组，如group1，从group1开始

    '''
    # 切分点从小到大排序
    cutoffs = sorted(cutoffs)
    num_groups = len(cutoffs)
    # 异常情况：小于第一组的起始值，这里直接放到第一组
    # 异常值建议在分组之前先处理妥善
    if x < cutoffs[0]:
        return 'group1'

    for i in range(1, num_groups):
        if cutoffs[i - 1] <= x < cutoffs[i]:
            return 'group{}'.format(i)
    # 最后一组，也可能会包括一些非常大的异常值
    return 'group{}'.format(num_groups)

