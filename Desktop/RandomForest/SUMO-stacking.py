import docx
from numpy import *
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from  sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
# import tensorflow as tf
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, \
            classification_report, recall_score, precision_recall_curve,accuracy_score,matthews_corrcoef
from mlxtend.classifier import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def getSite(filename):
    doc = docx.Document(filename)
    siteList = []
    for site in doc.paragraphs:
        siteList.append(site.text)
    #print(siteList)
    return siteList

# 判断是否为天然氨基酸，用X代表其他所有氨基酸，所有氨基酸用数字来替代，构造单个频率矩阵以及n_gram频率矩阵
def replace_no_native_amino_acid(lists, datatype):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    # num_list = []
    # print(len(lists))
    # print(len(lists[0]))
    # print(native_amino_acid[0])
    numlists = zeros((len(lists), 22))
    frequency_array = zeros((21, 21))
    n_gram_frequency_array = zeros((400, 20))
    flag = 1
    for site in range(len(lists)):
        for i in range(len(lists[0])):#位点位置
            for j in range(len(native_amino_acid)):#氨基酸种类
                # print(site, i, j)
                if lists[site][i] == native_amino_acid[j]:
                    numlists[site][i] = j+1
                    frequency_array[j][i] = frequency_array[j][i] + 1
                    flag = 0
                    if i > 0 & i < (len(lists[0])-2):
                        a = (numlists[site][i-1]-1) * 20
                        b = numlists[site][i] - 1
                        n_gram_index = int(a + b)
                        # print(a)
                        n_gram_frequency_array[n_gram_index][i-1] = n_gram_frequency_array[n_gram_index][i-1] + 1
                    break
            if flag != 0:
                # site = site[:i] + 'X' +site[i+1:]
                numlists[site][i] = 21
                frequency_array[20][i] = frequency_array[20][i]+1
            flag = 1
            if datatype == 1:
                numlists[site][21] = 1
            # print(i)
                # print(site)
        # replaced_list.append(site)
        # print(site)
    length = len(lists)
    for i in range(len(frequency_array)):
        for j in range(len(frequency_array[0])):
            frequency_array[i][j] =frequency_array[i][j]/length
    for i in range(len(n_gram_frequency_array)):
        for j in range(len(n_gram_frequency_array[0])):
            n_gram_frequency_array[i][j] = n_gram_frequency_array[i][j]/length
    # for r in n_gram_frequency_array:
    #     print(r)
    # print(frequency_array)
    return numlists, frequency_array, n_gram_frequency_array

# 构造skip_gram的频率矩阵
def get_skip_gram_frequency_array(lists, datatype, k):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    # print(len(lists))
    numlists = zeros((len(lists), 22))
    skip_gram_frequency_array = zeros((400, 20-k))
    flag = 1
    for site in range(len(lists)):
        for i in range(len(lists[0])):  # 位点位置
            for j in range(len(native_amino_acid)):  # 氨基酸种类
                # print(site, i, j)
                if lists[site][i] == native_amino_acid[j]:
                    numlists[site][i] = j + 1
                    flag = 0
                    if i > k:
                        a = (numlists[site][i - k - 1] - 1) * 20
                        b = numlists[site][i] - 1
                        skip_gram_index = int(a + b)
                        # print(a)
                        skip_gram_frequency_array[skip_gram_index][i - k - 1] = skip_gram_frequency_array[skip_gram_index][i-k-1] + 1
                    break
            if flag != 0:
                # site = site[:i] + 'X' +site[i+1:]
                numlists[site][i] = 21
            flag = 1
            if datatype == 1:
                numlists[site][21] = 1
                # print(i)
                # print(site)
                # replaced_list.append(site)
                # print(site)
    length = len(lists)
    for i in range(len(skip_gram_frequency_array)):
        for j in range(len(skip_gram_frequency_array[0])):
            skip_gram_frequency_array[i][j] = skip_gram_frequency_array[i][j] / length
    # for r in skip_gram_frequency_array:
    #     print(r)
    # print(frequency_array)
    return skip_gram_frequency_array


# exmplelist = ['AAAAAAAAAAKAAAAAAAAAA', 'CCCCCCCCCCKCCCCCCCCCC', 'AAAAAAAAAAKAAAAAAAAAA',
#               'AAAAAAAAAAKAAAAAAAAAA', 'AAAAAAAAAAKAAAAAAAAAA']
# skip_gram_frequency_array(exmplelist, 1, 1)

# 正样本频率减去负样本频率得到整体样本频率
def result_frequency_site(positive_site, negative_site,datatype):
    result_site = zeros((len(positive_site), len(positive_site[0])))
    for i in range(len(positive_site)):
        for j in range(len(positive_site[0])):
            if datatype == "frequency":
                result_site[i][j] = positive_site[i][j] - negative_site[i][j]
            elif datatype == "entropy":
                result_site[i][j] = negative_site[i][j] - positive_site[i][j]
    # print(result_site)
    return result_site

# 把序号表示矩阵转换到频率表示或熵表示
def to_site(lists, datatype, frequency_array):
    full_frequency_array = zeros((len(lists), 22))
    # print(len(lists))
    j = 0
    for site in range(len(lists)):
        # j = j+1
        for i in range(len(lists[0])-1):#位点位置
            # print(full_frequency_array[j][0])
            position = int(lists[site][i])
            # print(position)
            # print(shape(position))
            full_frequency_array[site][i] =frequency_array[position-1][i]
        if datatype == 1:#标记正负样本
            full_frequency_array[site][21] = 1
    # for r in full_frequency_array:
    #     if datatype ==1:
    #         print(r)
    #     # print(r)
    return full_frequency_array

# 把序号表示矩阵转换到n_gram频率表示
def to_n_gram_site(lists, datatype, n_gram_frequency_array):
    full_n_gram_frequency_array = zeros((len(lists), 21))
    # print(len(lists))
    for site in range(len(lists)):
        for i in range(len(lists[0])-1):#位点位置
            if i > 0:
                position_a = int(lists[site][i-1]) - 1
                position_b = int(lists[site][i]) - 1
                position_index = int(position_a * 20 + position_b)
                full_n_gram_frequency_array[site][i-1] = n_gram_frequency_array[position_index][i-1]
        if datatype == 1:#标记正负样本
            full_n_gram_frequency_array[site][20] = 1
    # for r in full_frequency_array:
    #     if datatype ==1:
    #         print(r)
    #     # print(r)
    return full_n_gram_frequency_array

# 把序号表示矩阵转换到skip_gram频率表示
def to_skip_gram_site(lists, datatype,skip_gram_frequency_array, k):
    full_skip_gram_frequency_array = zeros((len(lists), 21-k))
    # print(len(lists))
    for site in range(len(lists)):
        for i in range(len(lists[0])-1):#位点位置
            if i > k:
                position_a = int(lists[site][i-1-k]) - 1
                position_b = int(lists[site][i]) - 1
                position_index = int(position_a * 20 + position_b)
                full_skip_gram_frequency_array[site][i-1-k] = skip_gram_frequency_array[position_index][i-1-k]
        if datatype == 1:#标记正负样本
            arraylen = len(full_skip_gram_frequency_array[0])-1
            full_skip_gram_frequency_array[site][arraylen] = 1
    # for r in full_skip_gram_frequency_array:
    #     print(r)
    return full_skip_gram_frequency_array

# 算出每一列的熵
def entropy_of_site(arrays):
    entropy_arrays = zeros((len(arrays), 2))
    for site in range(len(arrays)):
        for i in range(len(arrays[0])-1):
            if arrays[site][i] == 0:
                break
            else:
                entropy_arrays[site][0] = entropy_arrays[site][0] - arrays[site][i] * np.math.log(arrays[site][i])
        entropy_arrays[site][1] = arrays[site][len(arrays[0])-1]
    # for r in arrays:
    #     print(r)
    return entropy_arrays

# # 算出每一列的序列特异性评分sequence specificity score,传入的是序号矩阵，正负频率分数，
# def specificity_score_of_site(arrays,positive_site,negative_site, datatype):
#     specificity_score_arrays = zeros((len(arrays), 2))
#     for i in range(len(arrays)):
#         for j in range(len(arrays[0])-1):
#             if positive_site[i][j] == 0 & negative_site[i][j] == 0:
#                 break
#             else:
#                 score_p = score_p + 0.5 * positive_array[i][j] * np.math.log(positive_array[i][j]/(positive_array[i][j]+negative_array[i][j]))
#                 score_n = score_n + 0.5 * negative_array[i][j] * np.math.log(negative_array[i][j]/(positive_array[i][j]+negative_array[i][j]))
#         specificity_score_arrays[i][0] = 0 - score_p - score_n
#         specificity_score_arrays[i][1] = datatype
#     return specificity_score_arrays

# 条件频率矩阵,传入的是整体的频率矩阵
def conditional_frequency(arrays):
    conditional_frequency_array = zeros((len(arrays), len(arrays[0])))
    for site in range(len(arrays)):
        for i in range(len(arrays[0])):
            # print(len(arrays[0]))
            if arrays[site][i] ==0:
                conditional_frequency_array[site][i] = 0
            else:
                if (i < 9) & (i >= 0):
                    conditional_frequency_array[site][i] = arrays[site][i] / (arrays[site][i + 1])
                elif (i > 11):
                    conditional_frequency_array[site][i] = arrays[site][i] / (arrays[site][i - 1])
        conditional_frequency_array[site][9] = arrays[site][9]
        conditional_frequency_array[site][10] = arrays[site][10]
        conditional_frequency_array[site][11] =arrays[site][11]
    # for r in conditional_frequency_array:
    #     print(r)
    return conditional_frequency_array


# #用（条件，信息）熵来代替频率
# def to_entropy(arrays):
#     entropy_array = zeros((len(arrays), 21))
#     for site in range(len(arrays)):
#         for i in range(len(arrays[0])):  # 位点位置
#             if arrays[site][i] == 0:
#                 entropy_array[site][i] = 0
#             else:
#                 entropy_array[site][i] = 0 - arrays[site][i] * np.math.log(arrays[site][i], 2)
#         # entropy_array[site][21] = arrays[site][21]仅仅只传入了特征矩阵的分布21*21
#     # for r in entropy_array:
#     #     print(r)
#     return entropy_array

# 根据氨基酸在—1和+2的出现的氨基酸的特性构造两组特征
def hydrophobic_position_array(allarrary, datatype):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    type1 = ('I', 'L', 'V',)
    # 0.4388
    type2 = ('A', 'F', 'M', 'P', 'W',)
    # -0.031
    type3 = ('G', 'Y',)
    # -0.0644
    # other：-0.3725
    type4 = ('D', 'E',)
    # 0.6287
    # -0.6299
    position1_array = zeros((len(allarrary), 3))
    for i in range(len(allarrary)-1):

        # print(allarrary[i][9])
        if allarrary[i][9] in type1:
            position1_array[i][0] = 0.4388
        elif allarrary[i][9] in type2:
            position1_array[i][0] = -0.031
        elif allarrary[i][9] in type3:
            position1_array[i][0] = -0.064
        else:
            position1_array[i][0] = -0.3725

        # print(allarrary[i][12])
        if allarrary[i][12] in type4:
            position1_array[i][1] = 0.6287
        else:
            position1_array[i][1] = -0.6299
        position1_array[i][2] = datatype
    # for r in position1_array:
    #     print(r)
    return position1_array

# 根据氨基酸在—1和+2的出现的氨基酸的特性构造两组特征（0，1表示法）
def hydrophobic_position_array0(allarrary, datatype):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    type1 = ('I', 'L', 'V',)
    # 0.4388
    type2 = ('A', 'F', 'M', 'P', 'W',)
    # -0.031
    type3 = ('G', 'Y',)
    # -0.0644
    # other：-0.3725
    type4 = ('D', 'E',)
    # 0.6287
    # -0.6299
    position1_array = zeros((len(allarrary), 6))
    for i in range(len(allarrary)-1):

        # print(allarrary[i][9])
        if allarrary[i][9] in type1:
            for j in range(4):
                if j == 3:
                    position1_array[i][j] = 1
                else:
                    position1_array[i][j] = 0
        elif allarrary[i][9] in type2:
            for j in range(4):
                if j == 2:
                    position1_array[i][j] = 1
                else:
                    position1_array[i][j] = 0
        elif allarrary[i][9] in type3:
            for j in range(4):
                if j == 1:
                    position1_array[i][j] = 1
                else:
                    position1_array[i][j] = 0
        else:
            for j in range(4):
                if j == 0:
                    position1_array[i][j] = 1
                else:
                    position1_array[i][j] = 0

        # print(allarrary[i][12])
        if allarrary[i][12] in type4:
            position1_array[i][4] = 0
        else:
            position1_array[i][4] = 1
        position1_array[i][5] = datatype
    # for r in position1_array:
    #     print(r)
    return position1_array

# 特征矩阵拼接
def splice_feature_array(feature_array_x, feature_array_y):
    sum_feature_array = zeros((len(feature_array_x), len(feature_array_x[0])+len(feature_array_y[0])-1))
    for site in range(len(sum_feature_array)):
        for i in range(len(feature_array_x[0])-1):
            sum_feature_array[site][i] = feature_array_x[site][i]
        for i in range(len(feature_array_x[0])-1, len(sum_feature_array[0])):
            sum_feature_array[site][i] = feature_array_y[site][i+1-len(feature_array_x[0])]
    return sum_feature_array

# 保证生成的随机状态一致
random_state = 2018
np.random.seed(random_state)

# 决策树预测
def tree_prediction(feature_data, result_data):
    feature_train, feature_test, result_train, result_test = train_test_split(feature_data, result_data, test_size=0.3)
    class_weight = dict({0: 20.5, 1: 1.5})
    clf = tree.DecisionTreeClassifier(random_state=random_state, max_depth=4, min_samples_leaf=6, min_samples_split=18,
                                      max_features=85, criterion='gini', class_weight=class_weight)
    # 利用网格搜索查找最佳参数
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = make_scorer(matthews_corrcoef)

    params = {
        # 'min_samples_split': range(2, 42, 4),
        # 'min_samples_leaf': range(2, 10, 2)
        'max_features': range(70, 90, 1)
    }

    grid_clf = GridSearchCV(estimator=clf, param_grid=params, cv=cv, n_jobs=-1, verbose=4, scoring=scoring)
    grid_clf.fit(feature_train, result_train.ravel())
    # 输出找到的最佳参数和分类器的组成
    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    print(grid_clf.best_score_)

# SVM预测(还没调出结果)
def SVM_prediction(feature_data, result_data):
    feature_train, feature_test, result_train, result_test = train_test_split(feature_data, result_data, test_size=0.3)

    svmf = SVC(gamma='auto', C=0.001, kernel="linear")
    # 利用网格搜索查找最佳参数
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = make_scorer(matthews_corrcoef)

    params = {
        # 'C': [0.001, 0.003, 0.006, 0.009, 0.01, 0.04, 0.08, 0.1],
        #       'kernel': ('linear', 'rbf',),
              # 'gamma': [0.001, 0.005, 0.1, 0.15, 0.20, 0.23, 0.27],
              # 'decision_function_shape': ['ovo', 'ovr']
                  }

    grid_clf = GridSearchCV(estimator=svmf, param_grid=params, cv=cv, n_jobs=-1, verbose=4, scoring=scoring)
    grid_clf.fit(feature_train, result_train.ravel())
    # 输出找到的最佳参数和分类器的组成
    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    print(grid_clf.best_score_)



# 利用随机森林进行预测
def RandomForest_prediction(feature_data, result_data):
    feature_train, feature_test, result_train, result_test = train_test_split(feature_data, result_data, test_size=0.2,
                                                                              random_state=random_state)

    # 利用网格搜索查找最佳参数
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    #
    # clf = RandomForestClassifier(max_depth=None, random_state=None, max_features=None, n_estimators=200, min_samples_leaf=100,
    #                              min_samples_split=100)#, obb_score=True) #class_weight="balanced")
    # scoring = make_scorer(matthews_corrcoef)
    #
    #
    # params = {
    #           'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    #           # 'min_samples_leaf': range(50, 100, 5)
    #           }
    #
    # grid_clf = GridSearchCV(estimator=gbc, param_grid=params, cv=cv, n_jobs=-1, verbose=4, scoring=scoring)
    # grid_clf.fit(feature_train, result_train.ravel())
    # # 输出找到的最佳参数和分类器的组成
    # print(grid_clf.best_estimator_)
    # print(grid_clf.best_params_)
    # print(grid_clf.best_score_)
    #
    # 联合集成学习的k折交叉验证
    class Create_ensemble(object):
        def __init__(self, n_splits, base_models):
            self.n_splits = n_splits
            self.base_models = base_models

        def predict(self, X, Y):
            X = np.array(X)
            Y = np.array(Y)

            folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=random_state).split(X, Y))
            train_pred = np.zeros((X.shape[0], len(self.base_models)))
            acc_scores = np.zeros((len(self.base_models), self.n_splits))
            recall_scores = np.zeros((len(self.base_models), self.n_splits))
            mcc_scores = np.zeros((len(self.base_models), self.n_splits))
            f1_scores = np.zeros((len(self.base_models), self.n_splits))
            for i, clf in enumerate(self.base_models):
                for j, (train_idx, valid_idx) in enumerate(folds):
                    X_train = X[train_idx]
                    Y_train = Y[train_idx]
                    X_valid = X[valid_idx]
                    Y_valid = Y[valid_idx]

                    clf.fit(X_train, Y_train)

                    valid_pred = clf.predict(X_valid)
                    recall = recall_score(Y_valid, valid_pred, average='macro')
                    f1 = f1_score(Y_valid, valid_pred, average='macro')
                    acc = accuracy_score(Y_valid, valid_pred)
                    mcc = matthews_corrcoef(Y_valid, valid_pred)

                    recall_scores[i][j] = recall
                    f1_scores[i][j] = f1
                    acc_scores[i][j] = acc
                    mcc_scores[i][j] = mcc

                    train_pred[valid_idx, i] = valid_pred

                    print(
                        "Model- {} and CV- {} recall: {}, acc_score: {} , mcc_score: {}".format(i, j, recall, acc, mcc))
            return train_pred, recall_scores, f1_scores

    # rdf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=8,
    #                              max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    #                              min_impurity_split=None, min_samples_leaf=4, min_samples_split=10,
    #                              min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1, oob_score=False,
    #                              random_state=random_state, verbose=0, warm_start=False)
    # rdf=GradientBoostingClassifier(random_state=random_state, learning_rate=0.2, n_estimators=261,
    #                                max_depth=10, min_samples_split=2, min_samples_leaf=95,
    #                                max_features=21, subsample=0.75)
    class_weight = dict({0: 20.5, 1: 1.5})
    rdf =RandomForestClassifier()
    base_models = [rdf]
    n_splits = 5
    lgb_stack = Create_ensemble(n_splits=n_splits, base_models=base_models)
    train_pred, recall_scores, f1_scores = lgb_stack.predict(feature_data, result_data)

    confmat = confusion_matrix(result_data, train_pred)
    sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
    sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
    print('1. The acc score of the model {}\n'.format(accuracy_score(result_data, train_pred)))
    print('2. The sp score of the model {}\n'.format(sp))
    print('3. The sn score of the model {}\n'.format(sn))
    print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result_data, train_pred)))
    print('9. The auc score of the model {}\n'.format(roc_auc_score(result_data, train_pred, average='macro')))
    print('6. The recall score of the model {}\n'.format(recall_score(result_data, train_pred, average='macro')))
    print('5. The F-1 score of the model {}\n'.format(f1_score(result_data, train_pred, average='macro')))
    print('7. Classification report \n {} \n'.format(classification_report(result_data, train_pred)))
    print('8. Confusion matrix \n {} \n'.format(confusion_matrix(result_data, train_pred)))


def GBDT_prediction(feature_data, result_data):
    feature_train, feature_test, result_train, result_test = train_test_split(feature_data, result_data, test_size=0.2, random_state=random_state)
    # 利用网格搜索查找最佳参数
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    #
    # gbc = GradientBoostingClassifier(random_state=random_state, learning_rate=0.2, n_estimators=261,
    #                                  max_depth=10, min_samples_split=2, min_samples_leaf=95,
    #                                  max_features=21, subsample=0.75)
    # scoring = make_scorer(matthews_corrcoef)
    #
    #
    # params = {
    #           'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    #           # 'min_samples_leaf': range(50, 100, 5)
    #           }
    #
    # grid_clf = GridSearchCV(estimator=gbc, param_grid=params, cv=cv, n_jobs=-1, verbose=4, scoring=scoring)
    # grid_clf.fit(feature_train, result_train.ravel())
    # # 输出找到的最佳参数和分类器的组成
    # print(grid_clf.best_estimator_)
    # print(grid_clf.best_params_)
    # print(grid_clf.best_score_)
    #
    # 联合集成学习的k折交叉验证
    class Create_ensemble(object):
        def __init__(self, n_splits, base_models):
            self.n_splits = n_splits
            self.base_models = base_models

        def predict(self, X, Y):
            X = np.array(X)
            Y = np.array(Y)

            folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=random_state).split(X, Y))
            train_pred = np.zeros((X.shape[0], len(self.base_models)))
            acc_scores = np.zeros((len(self.base_models), self.n_splits))
            recall_scores = np.zeros((len(self.base_models), self.n_splits))
            mcc_scores = np.zeros((len(self.base_models), self.n_splits))
            f1_scores = np.zeros((len(self.base_models), self.n_splits))
            for i, clf in enumerate(self.base_models):
                for j, (train_idx, valid_idx) in enumerate(folds):
                    X_train = X[train_idx]
                    Y_train = Y[train_idx]
                    X_valid = X[valid_idx]
                    Y_valid = Y[valid_idx]

                    clf.fit(X_train, Y_train)

                    valid_pred = clf.predict(X_valid)
                    recall = recall_score(Y_valid, valid_pred, average='macro')
                    f1 = f1_score(Y_valid, valid_pred, average='macro')
                    acc = accuracy_score(Y_valid, valid_pred)
                    mcc = matthews_corrcoef(Y_valid, valid_pred)

                    recall_scores[i][j] = recall
                    f1_scores[i][j] = f1
                    acc_scores[i][j] = acc
                    mcc_scores[i][j] = mcc

                    train_pred[valid_idx, i] = valid_pred

                    print("Model- {} and CV- {} recall: {}, acc_score: {} , mcc_score: {}".format(i, j, recall, acc, mcc))
            return train_pred, recall_scores, f1_scores

    # rdf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=8,
    #                              max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    #                              min_impurity_split=None, min_samples_leaf=4, min_samples_split=10,
    #                              min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1, oob_score=False,
    #                              random_state=random_state, verbose=0, warm_start=False)
    # rdf=GradientBoostingClassifier(random_state=random_state, learning_rate=0.2, n_estimators=261,
    #                                max_depth=10, min_samples_split=2, min_samples_leaf=95,
    #                                max_features=21, subsample=0.75)
    rdf= GradientBoostingClassifier()
    base_models = [rdf]
    n_splits = 5
    lgb_stack = Create_ensemble(n_splits=n_splits, base_models=base_models)
    train_pred, recall_scores, f1_scores = lgb_stack.predict(feature_data, result_data)

    confmat = confusion_matrix(result_data, train_pred)
    sn = confmat[1, 1]/(confmat[1, 0]+confmat[1, 1])
    sp = confmat[0, 0]/(confmat[0, 0]+confmat[0, 1])
    print('1. The acc score of the model {}\n'.format(accuracy_score(result_data, train_pred)))
    print('2. The sp score of the model {}\n'.format(sp))
    print('3. The sn score of the model {}\n'.format(sn))
    print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result_data, train_pred)))
    print('9. The auc score of the model {}\n'.format(roc_auc_score(result_data, train_pred, average='macro')))
    print('6. The recall score of the model {}\n'.format(recall_score(result_data, train_pred, average='macro')))
    print('5. The F-1 score of the model {}\n'.format(f1_score(result_data, train_pred, average='macro')))
    print('7. Classification report \n {} \n'.format(classification_report(result_data, train_pred)))
    print('8. Confusion matrix \n {} \n'.format(confusion_matrix(result_data, train_pred)))

def LDA_prediction(feature_data, result_data):
    feature_train, feature_test, result_train, result_test = train_test_split(feature_data, result_data, test_size=0.2, random_state=random_state)
    # 利用网格搜索查找最佳参数
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    gbc = GradientBoostingClassifier(random_state=random_state, learning_rate=0.2, n_estimators=261,
                                     max_depth=10, min_samples_split=2, min_samples_leaf=95,
                                     max_features=21, subsample=0.75)
    scoring = make_scorer(matthews_corrcoef)


    params = {
              'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
              # 'min_samples_leaf': range(50, 100, 5)
              }

    grid_clf = GridSearchCV(estimator=gbc, param_grid=params, cv=cv, n_jobs=-1, verbose=4, scoring=scoring)
    grid_clf.fit(feature_train, result_train.ravel())
    # 输出找到的最佳参数和分类器的组成
    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    print(grid_clf.best_score_)

def LR_prediction(feature_data, result_data):
    feature_train, feature_test, result_train, result_test = train_test_split(feature_data, result_data, test_size=0.2,
                                                                              random_state=random_state)

    # # 利用网格搜索查找最佳参数
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    # class_weight = dict({0: 20.5, 1: 1.5})
    # lr = LogisticRegression(penalty='l2', class_weight=class_weight)
    # scoring = make_scorer(matthews_corrcoef)
    #
    # params = {
    #     'C': [0.1, 0.3, 0.5, 0.7, 0.9, 1],
    #     'solver': ['lbfgs', 'newton-cg', 'sag']
    #     # 'min_samples_leaf': range(50, 100, 5)
    # }
    #
    # grid_clf = GridSearchCV(estimator=lr, param_grid=params, cv=cv, n_jobs=-1, verbose=4, scoring=scoring)
    # grid_clf.fit(feature_train, result_train.ravel())
    # # 输出找到的最佳参数和分类器的组成
    # print(grid_clf.best_estimator_)
    # print(grid_clf.best_params_)
    # print(grid_clf.best_score_)

    # 联合集成学习的k折交叉验证
    class Create_ensemble(object):
        def __init__(self, n_splits, base_models):
            self.n_splits = n_splits
            self.base_models = base_models

        def predict(self, X, Y):
            X = np.array(X)
            Y = np.array(Y)

            folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=random_state).split(X, Y))
            train_pred = np.zeros((X.shape[0], len(self.base_models)))
            acc_scores = np.zeros((len(self.base_models), self.n_splits))
            recall_scores = np.zeros((len(self.base_models), self.n_splits))
            mcc_scores = np.zeros((len(self.base_models), self.n_splits))
            f1_scores = np.zeros((len(self.base_models), self.n_splits))
            for i, clf in enumerate(self.base_models):
                for j, (train_idx, valid_idx) in enumerate(folds):
                    X_train = X[train_idx]
                    Y_train = Y[train_idx]
                    X_valid = X[valid_idx]
                    Y_valid = Y[valid_idx]

                    clf.fit(X_train, Y_train)

                    valid_pred = clf.predict(X_valid)
                    recall = recall_score(Y_valid, valid_pred, average='macro')
                    f1 = f1_score(Y_valid, valid_pred, average='macro')
                    acc = accuracy_score(Y_valid, valid_pred)
                    mcc = matthews_corrcoef(Y_valid, valid_pred)

                    recall_scores[i][j] = recall
                    f1_scores[i][j] = f1
                    acc_scores[i][j] = acc
                    mcc_scores[i][j] = mcc

                    train_pred[valid_idx, i] = valid_pred

                    print(
                        "Model- {} and CV- {} recall: {}, acc_score: {} , mcc_score: {}".format(i, j, recall, acc, mcc))
            return train_pred, recall_scores, f1_scores
    class_weight = dict({0: 10, 1: 1.5})
    rdf = LogisticRegression(random_state=random_state, C=0.9, solver='newton-cg', class_weight=class_weight)
    base_models = [rdf]
    n_splits = 5
    lgb_stack = Create_ensemble(n_splits=n_splits, base_models=base_models)
    train_pred, recall_scores, f1_scores = lgb_stack.predict(feature_data, result_data)

    confmat = confusion_matrix(result_data, train_pred)
    sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
    sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
    print('1. The acc score of the model {}\n'.format(accuracy_score(result_data, train_pred)))
    print('2. The sp score of the model {}\n'.format(sp))
    print('3. The sn score of the model {}\n'.format(sn))
    print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result_data, train_pred)))
    print('9. The auc score of the model {}\n'.format(roc_auc_score(result_data, train_pred, average='macro')))
    print('6. The recall score of the model {}\n'.format(recall_score(result_data, train_pred, average='macro')))
    print('5. The F-1 score of the model {}\n'.format(f1_score(result_data, train_pred, average='macro')))
    print('7. Classification report \n {} \n'.format(classification_report(result_data, train_pred)))
    print('8. Confusion matrix \n {} \n'.format(confusion_matrix(result_data, train_pred)))

#获取平均数
def Get_Average(list):
   sum = 0
   for item in list:
      sum += item
   return sum/len(list)

def Stacking(feature_data, result_data):
    feature_train, feature_test, result_train, result_test = train_test_split(feature_data, result_data, test_size=0.1, random_state=random_state)
    '''模型融合中使用到的各个单模型'''
    class_weight = dict({0: 1.5, 1: 20.5})
    clfs = [
            RandomForestClassifier(bootstrap=True, class_weight=class_weight, criterion='entropy', max_depth=15,
                                   max_features=40, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                   min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=174, n_jobs=-1, oob_score=False,
                                   random_state=random_state, verbose=0, warm_start=False),
            # tree.DecisionTreeClassifier(random_state=random_state, max_depth=4, min_samples_leaf=6, min_samples_split=18,
            #                             max_features=85, criterion='gini'),
            # KNeighborsClassifier(n_neighbors=50),
            # GaussianNB(),
            # MultinomialNB(), # 必须要求样本值为负，不符合
            # SVC(gamma='auto', C=0.001, kernel="linear",probability=True),
            # LogisticRegression(random_state=random_state, C=0.9, solver='newton-cg', class_weight=class_weight),
            GradientBoostingClassifier(random_state=random_state, learning_rate=0.2, n_estimators=261,
                                       max_depth=10, min_samples_split=2, min_samples_leaf=95,
                                       max_features=21, subsample=0.75)
    ]

    '''切分一部分数据作为测试集'''
    X, X_predict, y, y_predict = train_test_split(feature_data, result_data, test_size=0.2, random_state=random_state)
    skfall = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state).split(feature_data, result_data))
    acc = np.zeros(5).reshape(5,-1)
    sp = np.zeros(5).reshape(5,-1)
    sn = np.zeros(5).reshape(5,-1)
    mcc = np.zeros(5).reshape(5,-1)
    auc = np.zeros(5).reshape(5,-1)
    for k, (train, predict) in enumerate(skfall):
        X, y, X_predict, y_predict = feature_data[train], result_data[train], feature_data[predict], result_data[predict]
        dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
        dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))
        '''5折stacking'''
        n_splits = 5
        skf = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X, y))
        for j, clf in enumerate(clfs):
            '''依次训练各个单模型'''
            # print(j, clf)
            dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
            for i, (train, test) in enumerate(skf):
                '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
                # print("Fold", i)
                X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
                clf.fit(X_train, y_train)
                y_submission = clf.predict_proba(X_test)[:, 1]
                dataset_blend_train[test, j] = y_submission
                dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
            '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
            dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
            print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))
        # class_weightLR = dict({0: 0.8, 1: 1.8})
        class_weightLR= dict({0: 0.3, 1: 1.8})
        # clf = LogisticRegression(random_state=random_state, C=0.9, solver='newton-cg', class_weight=class_weight)
        clf = LogisticRegression()
        # clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
        clf.fit(dataset_blend_train, y)
        y_submission = clf.predict(dataset_blend_test)

        # print("Linear stretch of predictions to [0,1]")
        # y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
        print("blend result")
        print("val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))
        confmat = confusion_matrix(y_predict, y_submission)
        acc[k] = accuracy_score(y_predict, y_submission)
        sn[k] = (confmat[1, 1] / (confmat[1, 0] + confmat[1, 1]))
        sp[k] = (confmat[0, 0] / (confmat[0, 0] + confmat[0, 1]))
        mcc[k] = matthews_corrcoef(y_predict, y_submission)
        auc[k] = roc_auc_score(y_predict, y_submission, average='macro')
        # print('1. The acc score of the model {}\n'.format(accuracy_score(y_predict, y_submission)))
        # print('2. The sp score of the model {}\n'.format(sp[k]))
        # print('3. The sn score of the model {}\n'.format(sn[k]))
        # print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(y_predict, y_submission)))
        # print('9. The auc score of the model {}\n'.format(roc_auc_score(y_predict, y_submission, average='macro')))
        # print('6. The recall score of the model {}\n'.format(recall_score(y_predict, y_submission, average='macro')))
        # print('5. The F-1 score of the model {}\n'.format(f1_score(y_predict, y_submission, average='macro')))
        # print('7. Classification report \n {} \n'.format(classification_report(y_predict, y_submission)))
        # print('8. Confusion matrix \n {} \n'.format(confusion_matrix(y_predict, y_submission)))

        # class_weight = dict({0: 20.5, 1: 1.5})
        def re_predict(data, threshods):
            argmax = np.argmax(data)
            if argmax == 1:
                return argmax
            else:
                if data[argmax] >= threshods[argmax]:
                    return argmax
                else:
                    return (argmax + 1)

        y_submission_proba = clf.predict_proba(dataset_blend_test)

        threshold = [0.80, 0.30]
        new_pred = []
        print(y_submission.shape[0])
        for i in range(y_submission.shape[0]):
            new_pred.append(re_predict(y_submission_proba[i, :], threshold))
        print(k)
        confmat = confusion_matrix(y_predict, new_pred)
        # sp = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
        # # sn = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
        # acc[k] = accuracy_score(y_predict, new_pred)
        # sn[k] = (confmat[1, 1] / (confmat[1, 0] + confmat[1, 1]))
        # sp[k] = (confmat[0, 0] / (confmat[0, 0] + confmat[0, 1]))
        # mcc[k] = matthews_corrcoef(y_predict, new_pred)
        # auc[k] = roc_auc_score(y_predict, new_pred, average='macro')

        # print('1. The acc score of the model {}\n'.format(accuracy_score(y_predict, new_pred)))
        # print('2. The sp score of the model {}\n'.format(sn[k]))
        # print('3. The sn score of the model {}\n'.format(sp[k]))
        # print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(y_predict, new_pred)))
        # print('4. The auc score of the model {}\n'.format(roc_auc_score(y_predict, new_pred, average='macro')))
        # print('5. The F-1 score of the model {}\n'.format(f1_score(y_predict, new_pred, average='macro')))
        # print('6. The recall score of the model {}\n'.format(recall_score(y_predict, new_pred, average='macro')))
        # print('7. Classification report \n {} \n'.format(classification_report(y_predict, new_pred)))
        # print('8. Confusion matrix \n {} \n'.format(confusion_matrix(y_predict, new_pred)))
    aveacc = acc.mean()
    avesn = sn.mean()
    avesp = sp.mean()
    avemcc = mcc.mean()
    aveauc = auc.mean()
    print("--------------------------------------------------------------------------")
    print('1. The acc score of the model {}\n'.format(aveacc))
    print('2. The sp score of the model {}\n'.format(avesp))
    print('3. The sn score of the model {}\n'.format(avesn))
    print('4. The mcc score of the model {}\n'.format(avemcc))
    print('5. The auc score of the model {}\n'.format(aveauc))
    # print('5. The F-1 score of the model {}\n'.format(f1_score(y_predict, new_pred, average='macro')))
    # print('6. The recall score of the model {}\n'.format(recall_score(y_predict, new_pred, average='macro')))
    # print('7. Classification report \n {} \n'.format(classification_report(y_predict, new_pred)))
    # print('8. Confusion matrix \n {} \n'.format(confusion_matrix(y_predict, new_pred)))


if __name__ == '__main__':
    psiteList = getSite('PositiveData.docx')
    nsiteList = getSite('NegativeData.docx')
    positive = 1
    negative = 0
    positive_array, positive_frequency_site, positive_n_gram_frequency_site = replace_no_native_amino_acid(psiteList, positive)
    negative_array, negative_frequency_site, negative_n_gram_frequency_site = replace_no_native_amino_acid(nsiteList, negative)

    # print(positive_frequency_site)
    # print(negative_frequency_site)
    # print('wao')

    # 只利用序列编号进行预测
    allarray = np.concatenate((positive_array, negative_array), axis=0)
    # print(allarray)
    # # print(shape(allarray))
    # x, y = np.split(allarray, (21,), axis=1)

    # 利用频率矩阵进行预测
    sum_frequency_site = result_frequency_site(positive_frequency_site, negative_frequency_site, "frequency")
    positive_frequency_array = to_site(positive_array, positive, sum_frequency_site)
    negative_frequency_array = to_site(negative_array, negative, sum_frequency_site)
    frequency_allarray = np.concatenate((positive_frequency_array, negative_frequency_array), axis=0)
    print("frequency_allarray:"+str((len(frequency_allarray[0])-1)))
    # x, y = np.split(frequency_allarray, (21,), axis=1)

    # 利用n_gram频率矩阵进行预测
    sum_n_gram_frequency_site = result_frequency_site(positive_n_gram_frequency_site, negative_n_gram_frequency_site, "frequency")
    positive_n_gram_frequency_array = to_n_gram_site(positive_array, positive, sum_n_gram_frequency_site)
    negative_n_gram_frequency_array = to_n_gram_site(negative_array, negative, sum_n_gram_frequency_site)
    n_gram_frequency_allarray = np.concatenate((positive_n_gram_frequency_array, negative_n_gram_frequency_array), axis=0)
    # x, y =np.split(n_gram_frequency_allarray, (20,), axis=1)

    # 利用skip_gram频率矩阵进行预测 1，2，3
    positive_skip_gram_frequency_site1 = get_skip_gram_frequency_array(psiteList, positive, 1)
    negative_skip_gram_frequency_site1 = get_skip_gram_frequency_array(nsiteList, negative, 1)
    sum_skip_gram_frequency_site1 = result_frequency_site(positive_skip_gram_frequency_site1,
                                                         negative_skip_gram_frequency_site1, "frequency")
    positive_skip_gram_frequency_array1 = to_skip_gram_site(positive_array, positive, sum_skip_gram_frequency_site1, 1)
    negative_skip_gram_frequency_array1 = to_skip_gram_site(negative_array, negative, sum_skip_gram_frequency_site1, 1)
    skip_gram_frequency_allarray1 = np.concatenate(
        (positive_skip_gram_frequency_array1, negative_skip_gram_frequency_array1), axis=0)

    positive_skip_gram_frequency_site2 = get_skip_gram_frequency_array(psiteList, positive, 2)
    negative_skip_gram_frequency_site2 = get_skip_gram_frequency_array(nsiteList, negative, 2)
    sum_skip_gram_frequency_site2 = result_frequency_site(positive_skip_gram_frequency_site2,
                                                         negative_skip_gram_frequency_site2, "frequency")
    positive_skip_gram_frequency_array2 = to_skip_gram_site(positive_array, positive, sum_skip_gram_frequency_site2, 2)
    negative_skip_gram_frequency_array2 = to_skip_gram_site(negative_array, negative, sum_skip_gram_frequency_site2, 2)
    skip_gram_frequency_allarray2 = np.concatenate(
        (positive_skip_gram_frequency_array2, negative_skip_gram_frequency_array2), axis=0)

    positive_skip_gram_frequency_site3 = get_skip_gram_frequency_array(psiteList, positive, 3)
    negative_skip_gram_frequency_site3 = get_skip_gram_frequency_array(nsiteList, negative, 3)
    sum_skip_gram_frequency_site3 = result_frequency_site(positive_skip_gram_frequency_site3,
                                                         negative_skip_gram_frequency_site3, "frequency")
    positive_skip_gram_frequency_array3 = to_skip_gram_site(positive_array, positive, sum_skip_gram_frequency_site3, 3)
    negative_skip_gram_frequency_array3 = to_skip_gram_site(negative_array, negative, sum_skip_gram_frequency_site3, 3)
    skip_gram_frequency_allarray3 = np.concatenate(
        (positive_skip_gram_frequency_array3, negative_skip_gram_frequency_array3), axis=0)

    min_skip_gram_frequency_allarray=splice_feature_array(skip_gram_frequency_allarray1,skip_gram_frequency_allarray2)
    skip_gram_frequency_allarray=splice_feature_array(min_skip_gram_frequency_allarray,skip_gram_frequency_allarray3)
    bigram_feature_array = splice_feature_array(n_gram_frequency_allarray, skip_gram_frequency_allarray)
    print("bigram_feature_array:"+str(len(bigram_feature_array[0])-1))
    # x, y = np.split(bigram_feature_array, (len(bigram_feature_array[0])-1,), axis=1)

    # 利用条件频率矩阵进行预测
    positive_conditional_frequency_array = conditional_frequency(positive_frequency_array)
    negative_conditional_frequency_array = conditional_frequency(negative_frequency_array)
    conditional_frequency_array = np.concatenate((positive_conditional_frequency_array, negative_conditional_frequency_array), axis=0)
    print("conditional_frequency_array:"+str(len(conditional_frequency_array[0])-1))
    # x, y = np.split(conditional_frequency_array, (len(conditional_frequency_array[0])-1,), axis=1)
    # for r in conditional_frequency_array:
    #     print(r)

    sum_frequency_site = result_frequency_site(positive_frequency_site, negative_frequency_site, "frequency")
    positive_frequency_array = to_site(positive_array, positive, sum_frequency_site)
    negative_frequency_array = to_site(negative_array, negative, sum_frequency_site)
    frequency_allarray = np.concatenate((positive_frequency_array, negative_frequency_array), axis=0)
    print("frequency_allarray:"+str((len(frequency_allarray[0])-1)))
    # x, y = np.split(conditional_frequency_array, (21,), axis=1)
    # #  利用信息熵矩阵进行预测
    #    positive_entropy_site = to_entropy(positive_frequency_site)
    # print("正熵样本")
    # for r in positive_entropy_site:
    #     print(r)
    # print("负熵样本")
    # negative_entropy_site = to_entropy(negative_frequency_site)
    #   for r in negative_entropy_site:
    #     print(r)
    # sum_entropy_site = result_frequency_site(positive_entropy_site, negative_entropy_site, "entropy")
    #   print("总样本")
    # for r in sum_entropy_site:
    #     print(r)
    # positive_entropy_array = to_site(positive_array, 1, negative_entropy_site)
    # negative_entropy_array = to_site(positive_array, 0, negative_entropy_site)
    # entropy_allarray = np.concatenate((positive_entropy_array, negative_entropy_array), axis=0)
    # x, y = np.split(entropy_allarray, (21,), axis=1)
    # # 效果也太差了吧， 还比不上掷色子呢

    # 单单只利用每一列得到的信息熵进行预测
    only_positive_frequency_array = to_site(positive_array, 1, positive_frequency_site)
    only_negative_frequency_array = to_site(negative_array, 0, negative_frequency_site)
    positive_entropy_array = entropy_of_site(only_positive_frequency_array)
    negative_entropy_array = entropy_of_site(only_negative_frequency_array)
    entropy_allarray = np.concatenate((positive_entropy_array, negative_entropy_array), axis=0)
    # x, y = np.split(entropy_allarray, (1,), axis=1)

    # 使用每一列得到的条件熵进行预测
    only_positive_conditional_frequency_array = conditional_frequency(only_positive_frequency_array)
    only_negative_conditional_frequency_array = conditional_frequency(only_negative_frequency_array)
    positive_conditional_entropy_array = entropy_of_site(only_positive_conditional_frequency_array)
    negative_conditional_entropy_array = entropy_of_site(only_negative_conditional_frequency_array)
    conditional_entropy_allarray = np.concatenate((positive_conditional_entropy_array, negative_conditional_entropy_array), axis=0)
    # for r in conditional_entropy_allarray:
    #     print(r)
    # x, y = np.split(conditional_entropy_allarray, (1,), axis=1)

    # 信息熵加上条件熵进行预测
    conditional_and_entropy_allarray = splice_feature_array(conditional_entropy_allarray, entropy_allarray)
    print("conditional_and_entropy_allarray:"+str(len(conditional_and_entropy_allarray[0])-1))
    # for r in conditional_and_entropy_allarray:
    #       print(r)
    # x, y = np.split(conditional_and_entropy_allarray, (len(conditional_and_entropy_allarray[0])-1, ), axis=1)

    # 联合频率矩阵以及信息熵和条件熵进行预测(改成了只包含信息熵)
    conditional_allarray = splice_feature_array(conditional_frequency_array, entropy_allarray)
    # x, y = np.split(conditional_allarray, (len(conditional_allarray[0])-1, ), axis=1)
    # # 再加上矩阵特异性分数 sequence specificity score(先放一边等会再加)
    # positive_specificity_array = specificity_score_of_site(only_positive_frequency_array, only_negative_frequency_array, 1)
    # negative_specificity_array = specificity_score_of_site(only_positive_frequency_array, only_negative_frequency_array, 0)




    # two_feature_array = splice_feature_array(frequency_allarray, conditional_frequency_array)
    three_feature_array =splice_feature_array(conditional_frequency_array, frequency_allarray)
    # for r in union_frequency_entropy_array:
    #     print(r)
    # x, y = np.split(union_frequency_entropy_array, (len(union_frequency_entropy_array[0])-1, ), axis=1)

    # 再加上位置特征集进行预测
    positive_position_array = hydrophobic_position_array0(psiteList, 1)
    negative_position_array = hydrophobic_position_array0(nsiteList, 0)
    position_array = np.concatenate((positive_position_array, negative_position_array), axis=0)
    print("position_array:"+str(len(position_array[0])-1))
    four_feature_array = splice_feature_array(three_feature_array, position_array)
    # x, y = np.split(position_array, (len(position_array[0])-1, ), axis=1)
    # x, y = np.split(four_feature_array, (len(four_feature_array[0])-1, ), axis=1)
    # 加上n_gram进行预测
    five_feature_array = splice_feature_array(position_array, n_gram_frequency_allarray)
    # x, y =np.split(all_feature_array, (len(five_feature_array[0])-1, ), axis=1)

    # 再加上skip_gram进行预测
    six_feature_array = splice_feature_array(five_feature_array, skip_gram_frequency_allarray)
    x, y = np.split(six_feature_array, (len(six_feature_array[0])-1, ), axis=1)
    # print(shape(x))
    # y.ravel('F')
    # tree_prediction(x, y.ravel())
    # RandomForest_prediction(x, y.ravel())
    # BP_prediction(x,y)
    Stacking(x, y.ravel())
    # GBDT_prediction(x, y.ravel())
    # SVM_prediction(x, y.ravel())
    # LR_prediction(x, y.ravel())



