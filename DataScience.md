# 数据科学比赛一般流程

- [数据科学比赛一般流程](#数据科学比赛一般流程)
  - [数据可视化](#数据可视化)
  - [特征工程](#特征工程)
  - [模型选择](#模型选择)

首先安装依赖：
```bash
pip install numpy matplotlib pandas seaborn sklearn gplearn xgboost pydotplus
```

## 数据可视化

一般是绘制分布图、箱线图、相关系数热力图等等：
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set()

# 分布图
def draw_hist(data):
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    for idx, col in enumerate(data.columns):
        if col == 'label':
            continue
        sns.histplot(data=data, x=col, ax=axes[idx//5, idx%5], 
                    kde=True, stat="probability", hue='label',
                    element="bars", common_norm=False, bins=50)
        #sns.histplot(data=data, x=col, ax=axes[idx//5, idx%5], 
        #            kde=True, stat="probability", hue='label',
        #            element="bars", common_norm=False, discrete=True)
    plt.tight_layout()
    plt.savefig('distribution.png')
    plt.clf()


# 箱线图
def draw_box(data):
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    for idx, col in enumerate(data.columns):
        if col == 'label':
            continue
        sns.boxplot(y=col, data=data, hue='label', ax=axes[idx//5, idx%5],
                    showfliers=True, width=.5, gap=.2)
    plt.tight_layout()
    plt.savefig('difference.png')
    plt.clf()


# 相关系数热力图
def draw_corr(data):
    sns.set_context({"figure.figsize":(10,10)}) # resize to get full features
    sns.heatmap(data=data.corr(), square=True, cmap='RdBu_r')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig('corr.png')
    plt.clf()


# t-sne散点图
def draw_neighbor(data):
    labels = data['label'].values
    data = data.drop(['label'], axis=1).values

    # normalize
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # t-sne
    tsne = TSNE(n_components=2, learning_rate=500, n_iter=10000, metric='l1')
    data = tsne.fit_transform(data) 

    # plot
    cmap = plt.cm.Spectral
    plt.figure(figsize=(10, 10))
    for i in range(2):
        indices = labels == i
        plt.scatter(data[indices, 0], data[indices, 1], color=cmap(i/1.1), label=i)
    plt.legend()
    plt.tight_layout()
    plt.savefig('neighbor.png')    
```

树分类器特征重要性和树结构可视化：
```python
from xgboost import plot_tree, plot_importance

def _plot_importance(classifier):
    plot_importance(classifier, max_num_features=32, 
                    xlabel='# of occurrences in trees',
                    title='', importance_type='weight',
                    values_format='{v:.0f}', xlim=(0,2500))
    plt.tight_layout()
    plt.savefig('xgboost_feature_weight.png')
    plt.clf()

    plot_importance(classifier, max_num_features=32, 
                    xlabel='performance gain',
                    title='', importance_type='gain',
                    values_format='{v:.2f}', xlim=(0,30))
    plt.tight_layout()
    plt.savefig('xgboost_feature_gain.png')
    plt.clf()


def _plot_tree(classifier, i):
    plot_tree(classifier, num_trees=i-1)    # 画出第i棵树
    fig = plt.gcf()
    fig.set_size_inches(50, 6)
    plt.tight_layout()
    plt.savefig(f'tree_{i}.png')
    plt.clf()
```

SymbolicTransformer符号树的可视化：
```python
from gplearn.genetic import SymbolicTransformer
import pydotplus

st = SymbolicTransformer(...)
st.fit(X_train, y_train)

print(st)   # all equations

for i in range(10):
    graph = st._best_programs[i].export_graphviz()
    graph = pydotplus.graphviz.graph_from_dot_data(graph)
    graph.write_png(f'equation_{i}.png')
```

ROC曲线绘制：
```python
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def draw_roc(classifier):
    X_train, Y_train = load_data('train')
    classifier.fit(X_train, Y_train)
    
    X_dev, Y_dev = load_data('dev')
    try:
        Y_dev_pred = classifier.predict_proba(X_dev)[:, 1]
    except:
        try:
            Y_dev_pred = classifier.decision_function(X_dev)
        except:
            raise ValueError('classifier not support predict_proba or decision_function')

    fpr, tpr, _ = roc_curve(Y_dev, Y_dev_pred)
    
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig('roc.png')


# cross validation
from sklearn.model_selection import cross_val_predict, StratifiedKFold
def draw_roc_cross_validation():
    features, labels = load_data()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_probas = cross_val_predict(classifier, features, labels, cv=kf, 
                                 method='predict_proba', n_jobs=16, verbose=1)
    
    # 初始化 ROC 曲线的平均值
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    kf_splits = list(kf.split(features, labels))

    # 计算每一折的 ROC 曲线，然后将它们加到平均值中
    for train, test in kf_splits:
        fpr, tpr, _ = roc_curve(labels[test], y_probas[test][:, 1])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)

    # 计算平均值
    mean_tpr /= len(kf_splits)

    plt.plot(mean_fpr, mean_tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig('roc.png')
```

## 特征工程

通用的特征工程方法主要包括分布变换、特征分箱、特征挖掘、特征过滤等：
```python
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from gplearn.genetic import SymbolicTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def load_data():
    full_df = {}
    for split in ['train', 'dev', 'test']:
        with open(f'{split}.csv', 'r') as f:
            dataset = pd.read_csv(f)
        
        # transform distribution
        col_name = [...]    # columns to transform
        for col in col_name:
            dataset[col] = np.log1p(dataset[col])

        full_df[split] = dataset
    return full_df

#利用决策树获得最优分箱的边界值列表
def optimal_binning_boundary(x, y, max_bins, min_x, max_x):
    x = x.values  
    y = y.values
    
    clf = DecisionTreeClassifier(criterion='gini',
                                 max_leaf_nodes=max_bins,
                                 random_state=42)

    clf.fit(x.reshape(-1, 1), y)
    
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    
    boundary = []
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  
            boundary.append(threshold[i])

    boundary.sort()
    boundary = [min_x] + boundary + [max_x]
    return boundary

def split_into_bins(full_df):
    # merge train, dev, test
    all_data = pd.concat([full_df['train'], full_df['dev']], axis=0, ignore_index=True)
    all_data_with_test = pd.concat([full_df['train'], full_df['dev'], full_df['test']], axis=0, ignore_index=True)

    # split into bins
    col_name = [...]    # columns to split
    for col in col_name:
        min_value = all_data_with_test[col].min()-0.1
        max_value = all_data_with_test[col].max()+0.1
        bins = optimal_binning_boundary(all_data[col], all_data['label'], 20, min_value, max_value)
        #print(col, bins)
        for split in ['train', 'dev', 'test']:
            full_df[split][col] = pd.cut(full_df[split][col], bins=bins, labels=[i for i in range(len(bins)-1)], right=True).astype(np.int64)
    return full_df

def feature_mining(full_df):
    scalar = StandardScaler()
    st = SymbolicTransformer(
        generations=20,
        population_size=1000,
        hall_of_fame=100,
        n_components=10,
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'max', 'min'],
        parsimony_coefficient='auto',
        max_samples=0.8,
        metric='spearman',
        verbose=1,
        random_state=42,
        n_jobs=16
    )
    model = RandomForestClassifier(random_state=42)
    rfe = RFE(model, n_features_to_select=32, verbose=1)

    # merge train & dev
    all_data = pd.concat([full_df['train'], full_df['dev']], axis=0, ignore_index=True)
    X_train = all_data.drop(['label'], axis=1).values.copy().astype(np.float64)
    Y_train = all_data['label'].values.copy().astype(np.int32)
    print(X_train.shape, Y_train.shape)
    # normalize
    X_train = scalar.fit_transform(X_train)
    # mine features by genetic programming
    X_mined_train = st.fit_transform(X_train, Y_train)
    print(X_mined_train.shape)
    X_train = np.concatenate((X_train, X_mined_train), axis=1)
    print(X_train.shape)
    # remove unimportant features
    X_train = rfe.fit_transform(X_train, Y_train) 
    print(X_train.shape)

    new_df = {}
    for split in ['train', 'dev', 'test']:
        _features = full_df[split].drop(['label'], axis=1)
        labels = full_df[split]['label']

        features_value = scalar.transform(_features.values.astype(np.float64))
        features = pd.DataFrame(features_value, columns=_features.columns)

        mined_features_value = st.transform(features_value)
        mined_features = pd.DataFrame(mined_features_value, 
                                      columns=[f'mined_feature_{i}' for i in range(mined_features_value.shape[1])])
        
        features = pd.concat([features, mined_features], axis=1)
        features = features[features.columns[rfe.support_]]
        print(features.shape)

        final_features = pd.concat([features, labels], axis=1)
        new_df[split] = final_features

    return new_df

def save(new_df):
    for split in ['train', 'dev', 'test']:
        new_df[split].to_csv(f'{split}_new.csv', index=False)
```

## 模型选择

常见的机器学习模型在sklearn中都有实现，可以直接调用：
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def load_data(split):
    with open(f'./{split}_new.csv', 'r') as f:
        dataset = pd.read_csv(f)
    features = dataset.drop(['label'], axis=1).values
    labels = dataset['label'].values
    return features, labels


def gridsearch(classifier, param_grid):
    # read data
    X_train , Y_train = load_data('train')

    # grid search
    grid_search = GridSearchCV(classifier, param_grid, scoring='accuracy', \
                               cv=5, n_jobs=16, verbose=1)
    grid_search.fit(X_train, Y_train)
    print(f'Best Grid parameters On Train: {grid_search.best_params_}')
    print(f'Best Grid score On Train: {grid_search.best_score_}')

    return grid_search.best_params_


def run(classifier):
    X_train, Y_train = load_data('train')
    print(X_train.shape, Y_train.shape)
    
    classifier.fit(X_train, Y_train)
    Y_train_pred = classifier.predict(X_train)
    print('On train set:')
    print(accuracy_score(Y_train, Y_train_pred))
    
    X_dev, Y_dev = load_data('dev')
    Y_dev_pred = classifier.predict(X_dev)
    print('On dev set:')
    print(accuracy_score(Y_dev, Y_dev_pred))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, help='algorithm name')
    args = parser.parse_args()

    if args.algo == 'svm':
        classifier = SVC(random_state=42)
        param_grid = {
            'C': [0.1, 0.2, 0.5, 0.8, 1, 2],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1, 2]
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'C': 0.8, 'kernel': 'linear', 'random_state': 42}
        run(SVC(**best_params_))
    elif args.algo == 'dt':
        classifier = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [i for i in range(2, 21, 2)],
            'min_samples_split': [i for i in range(2, 22, 2)],
            'min_samples_leaf': [i for i in range(1, 11, 1)],
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'criterion': 'gini', 'max_depth': 10, \
                       'max_features': 'sqrt', 'min_samples_leaf': 5, \
                       'min_samples_split': 10, 'random_state': 42}
        run(DecisionTreeClassifier(**best_params_))
    elif args.algo == 'xgb':
        classifier = XGBClassifier(objective='binary:logistic', \
                                subsample=0.8, colsample_bytree=0.8, \
                                random_state=42, n_jobs=16)
        param_grid = {
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6, 7],
            'n_estimators': [500, 800, 1000, 1200, 1500, 2000],
            'min_child_weight': [0.5, 0.7, 0.9, 1, 1.1],
            'gamma': [0, 0.01, 0.05, 0.1, 0.2, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.1, 0.5, 1]
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'learning_rate': 0.01, 'n_estimators': 1000, \
                        'max_depth': 5, 'gamma': 0.05, 'min_child_weight': 0.9, \
                        'reg_alpha': 0.1, 'reg_lambda': 0.5, \
                        'objective': 'binary:logistic', \
                        'subsample': 0.8, 'colsample_bytree': 0.8, \
                        'random_state': 42, 'n_jobs': 16}
        run(XGBClassifier(**best_params_))
    elif args.algo == 'rf':
        classifier = RandomForestClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [i for i in range(2, 21, 2)],
            'min_samples_split': [i for i in range(2, 22, 2)],
            'min_samples_leaf': [i for i in range(1, 11, 1)],
            'n_estimators': [i for i in range(50, 151, 10)],
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'criterion': 'gini', 'max_depth': 10, \
                       'max_features': 'sqrt', 'n_estimators': 150, \
                       'min_samples_leaf': 5, 'min_samples_split': 10, \
                       'random_state': 42}
        run(RandomForestClassifier(**best_params_))
    elif args.algo == 'nb':
        classifier = GaussianNB()
        run(classifier)
    elif args.algo == 'lr':
        classifier = LogisticRegression(random_state=42, n_jobs=16)
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.1, 0.2, 0.5, 0.8, 1, 2],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [500, 1000, 2000, 3000]
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'C': 2, 'max_iter': 500, 'penalty': 'l2', 'solver': 'saga', \
                        'random_state': 42, 'n_jobs': 16}
        run(LogisticRegression(**best_params_))
    else:
        print('Invalid algorithm name!')
```

有时还需要使用交叉验证：
```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def run(classifier):
    features, labels = load_data()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    score = cross_val_score(classifier, features, labels, 
                            cv=kf, scoring='accuracy', n_jobs=16)
```
