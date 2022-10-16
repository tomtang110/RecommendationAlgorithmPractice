
from sklearn.tree import DecisionTreeClassifier
def optimal_binning_boundary(x, y):
    '''
        利用决策树获得最优分箱的边界值列表,利用决策树生成的内部划分节点的阈值，作为分箱的边界
    '''
    boundary = []  # 待return的分箱边界值列表

    x = x.fillna(-1).values  # 填充缺失值
    y = y.values

    clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                 max_leaf_nodes=6,  # 最大叶子节点数
                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比

    clf.fit(x, y)  # 训练决策树

    # tree.plot_tree(clf) #打印决策树的结构图
    # plt.show()

    n_nodes = clf.tree_.node_count  # 决策树的节点数
    children_left = clf.tree_.children_left  # node_count大小的数组，children_left[i]表示第i个节点的左子节点
    children_right = clf.tree_.children_right  # node_count大小的数组，children_right[i]表示第i个节点的右子节点
    threshold = clf.tree_.threshold  # node_count大小的数组，threshold[i]表示第i个节点划分数据集的阈值

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 非叶节点
            boundary.append(threshold[i])

    boundary.sort()

    min_x = x.min()
    max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]

    return boundary