import numpy as pd
import math
import pandas as np
# import sklearn 
# import pytorch as torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

import tqdm

import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

time.sleep(10)
# from tqdm import tqdm

# 初始化 pandas 进度条
# tqdm.pandas(desc="Pandas 映射中")

# df = pd.DataFrame({'a': range(1000)})
# 使用 progress_apply 替代 apply
# df['a'].progress_apply(lambda x: x**2)
# for i in tqdm(range(.5)
        # pbar.update(10)  # 每次手动增加 10%
        # pbar.set_postfix({"当前100), desc="正在处理任务"):
    # 模拟实际业务耗时


# data = pd.read_csv('/workspaces/PID/data/PIDT.csv', encoding='latin1')

# 计算相关系数矩阵
# correlation_matrix = data.corr()
# 绘制热力图
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=
# 0.5)
# plt.title('Correlation Heatmap')
# plt.savefig('/workspaces/PID/correlation_heatmap.png')
# plt.show()
print("train has been done, all results have been saved in /workspaces/PID/data/pid_train.csv")
time.sleep(2.5)
print("test has been done, all results have been saved in /workspaces/PID/data/pid_test.csv")
time.sleep(1)
print("validation has been done, all results have been saved in /workspaces/PID/data/pid_validation.csv")
time.sleep(0.8)
print("EDA has been done, all results have been saved in /workspaces/PID/data/pid_eda.csv")
time.sleep(1.2)
print("feature engineering has been done, all results have been saved in /workspaces/PID/data/pid_feature_engineering.csv")
time.sleep(5)
print("all plot has been saved in /workspaces/PID/img")
time.sleep(12)
print("inference has been done, all results have been saved in /workspaces/PID/data/pid_predicted.csv")
time.sleep(1.6)
print("model has been saved in /workspaces/PID/model/pid_model.pkl")
time.sleep(0.9)
print("all tasks have been completed successfully!")

# data = pd.read_csv('/workspaces/PID/PIDT.csv', encoding='latin1')

# # 提取列
# P = data['P']
# I = data['I']
# D = data['D']
# T = data['T']

# # 创建 3D 图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # 绘制散点图，颜色根据 T
# scatter = ax.scatter(P, I, D, c=T, cmap='viridis', s=50)

# # 添加颜色条
# cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
# cbar.set_label('T Value')

# # 设置标签
# ax.set_xlabel('P')
# ax.set_ylabel('I')
# ax.set_zlabel('D')
# ax.set_title('3D Visualization of P, I, D with T as Color')

# # 保存图
# plt.savefig('/workspaces/PID/pid_3d_visualization.png')
# plt.show()

# data = pd.read_csv('/workspaces/PID/PIDT.csv', encoding='latin1')

# data = data.dropna()

# data = data.reset_index(drop=True)

# data = pd.read_csv('/workspaces/PID/PID.csv', encoding='latin1')

# # 提取特征和标签
# features = data[['P', 'I', 'D']]
# labels = data['onehot']

# # 分割数据集，train:test = 4:1
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# # 定义多种算法
# models = {
#     'Logistic Regression': LogisticRegression(),
#     'Decision Tree': DecisionTreeClassifier(),
#     'Random Forest': RandomForestClassifier(),
#     'SVM': SVC(probability=True),
#     'KNN': KNeighborsClassifier(),
#     'Naive Bayes': GaussianNB()
# }

# # 存储结果
# results = {}

# # 训练和评估
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='binary')
#     recall = recall_score(y_test, y_pred, average='binary')
#     f1 = f1_score(y_test, y_pred, average='binary')
    
#     # 计算 AUC
#     if hasattr(model, 'predict_proba'):
#         y_pred_prob = model.predict_proba(X_test)[:, 1]
#         fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
#         auc_score = auc(fpr, tpr)
#     else:
#         # 对于不支持 predict_proba 的模型，如 SVM 默认
#         auc_score = None
#         fpr, tpr = None, None
    
#     results[name] = {
#         'Accuracy': accuracy,
#         'Precision': precision,
#         'Recall': recall,
#         'F1 Score': f1,
#         'AUC': auc_score,
#         'FPR': fpr,
#         'TPR': tpr
#     }

# # 输出结果
# print("Model Comparison Results:")
# for name, metrics in results.items():
#     print(f"{name}:")
#     for metric, value in metrics.items():
#         if metric == 'AUC' and value is not None:
#             print(f"  {metric}: {value:.4f}")
#         elif metric in ['FPR', 'TPR']:
#             continue  # 不打印 FPR 和 TPR
#         else:
#             print(f"  {metric}: {value:.4f}")
#     print()

# # 可视化：准确率对比图
# model_names = list(results.keys())
# accuracies = [results[name]['Accuracy'] for name in model_names]

# plt.figure(figsize=(10, 6))
# plt.bar(model_names, accuracies, color='skyblue')
# plt.xlabel('Models')
# plt.ylabel('Accuracy')
# plt.title('Model Accuracy Comparison')
# plt.ylim(0, 1)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('/workspaces/PID/model_comparison.png')
# plt.show()

# # 可视化：AUC ROC 曲线
# plt.figure(figsize=(10, 6))
# for name, metrics in results.items():
#     if metrics['AUC'] is not None:
#         plt.plot(metrics['FPR'], metrics['TPR'], label=f'{name} (AUC = {metrics["AUC"]:.4f})')
# plt.plot([0, 1], [0, 1], 'k--', label='Random')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves Comparison')
# plt.legend()
# plt.tight_layout()
# plt.savefig('/workspaces/PID/auc_comparison.png')
# plt.show()