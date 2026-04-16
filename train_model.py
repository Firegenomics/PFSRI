import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, label_binarize
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from numpy import interp

def train_model(dataframe: pd.DataFrame, output_dir: str, plot_title: str):
    """训练模型"""
    output_dir = output_dir if output_dir.endswith('/') else output_dir + '/'
    dataframe_raw = dataframe.copy(deep=True)

    dataframe = dataframe[['Sample', 'Control_sample', 'HetRR', 'Relationship']]
    dataframe = dataframe.dropna(subset=['HetRR'])
    selected_classes = ['Identical', 'No significant relationship', 'Full-sibling']
    dataframe = dataframe[dataframe['Relationship'].isin(selected_classes)]

    train_data = dataframe.copy()
    X = train_data.drop(['Sample', 'Control_sample', 'Relationship'], axis=1)
    y = train_data['Relationship']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 定义模型
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, C=0.1),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=3),
        'SVM': SVC(random_state=42, probability=True, C=0.1)
    }
    
    # 定义模型绘图颜色
    model_colors = {
        'Logistic Regression': "#4C72B0",
        'Naive Bayes': "#55A868",
        'Decision Tree': "#DD8452",
        'SVM': "#E24A33"
    }

    k = 5
    model_accuracies = {}
    cv_results = {}
    
    # 1. 基础准确率交叉验证 (用于选择最佳模型)
    for model_name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=k)
        model_accuracies[model_name] = scores.mean()
        cv_results[model_name] = scores

    # 绘制 k 折交叉验证准确率结果图
    plt.figure(figsize=(12, 8))
    for model_name, scores in cv_results.items():
        plt.plot(range(1, k + 1), scores, marker='o', linestyle='-', label=model_name, color=model_colors[model_name])
    plt.ylim(0, 1.1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(range(1, k + 1))
    plt.xlabel('Fold', fontsize=14)
    plt.ylabel('Accuracy (K-fold Cross-Validation)', fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_dir + 'k_fold_cv_results.png', bbox_inches='tight')
    plt.close()

    best_model_name = max(model_accuracies, key=model_accuracies.get)
    best_model = models[best_model_name]
    print(f"Best Model Selected: {best_model_name}")

    # 2. ========== 新增：5 折交叉验证 ROC 曲线 (针对最佳模型) ==========
    # 使用 StratifiedKFold 确保每折类别分布一致
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # 用于存储每折的 ROC 数据 (One-vs-Rest)
    # 结构：cv_roc_data[class_idx] = [{'fpr':..., 'tpr':..., 'specificity':...}, ...] (k folds)
    cv_roc_data = {i: [] for i in range(n_classes)}
    mean_tpr_over_fpr = {i: [] for i in range(n_classes)} # 用于计算平均曲线
    mean_fpr = np.linspace(0, 1, 100) # 统一的 FPR 网格用于插值

    # 对训练集进行 5 折交叉验证以生成 ROC
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # 训练最佳模型
        # 注意：这里需要重新实例化模型，因为 sklearn 模型 fit 后不能再次 fit
        if best_model_name == 'Logistic Regression':
            fold_model = LogisticRegression(random_state=42, C=0.1)
        elif best_model_name == 'Naive Bayes':
            fold_model = GaussianNB()
        elif best_model_name == 'Decision Tree':
            fold_model = DecisionTreeClassifier(random_state=42, max_depth=3)
        elif best_model_name == 'SVM':
            fold_model = SVC(random_state=42, probability=True, C=0.1)
            
        fold_model.fit(X_fold_train, y_fold_train)
        
        # 获取预测概率
        if hasattr(fold_model, "predict_proba"):
            y_score = fold_model.predict_proba(X_fold_val)
        else:
            # 如果模型不支持 predict_proba，使用决策函数
            y_score = fold_model.decision_function(X_fold_val)
            if len(y_score.shape) == 1:
                y_score = y_score.reshape(-1, 1)

        # 对每个类别计算 ROC
        y_fold_val_bin = label_binarize(y_fold_val, classes=np.arange(n_classes))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_fold_val_bin[:, i], y_score[:, i])
            specificity = 1 - fpr  # 计算特异性
            
            # 保存当前折的数据
            cv_roc_data[i].append({
                'fpr': fpr,
                'tpr': tpr,
                'specificity': specificity,
                'auc': auc(fpr, tpr)
            })
            
            # 插值以计算平均曲线 (在统一的 FPR 网格上)
            interp_tpr = interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            mean_tpr_over_fpr[i].append(interp_tpr)

    # 绘制 5 折 CV ROC 图 (每个类别一张图)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.0

    for i in range(n_classes):
        class_label = le.classes_[i]
        plt.figure(figsize=(6, 6))
        
        # 绘制 5 折的个体曲线
        for fold_idx in range(k):
            data_fold = cv_roc_data[i][fold_idx]
            plt.plot(data_fold['specificity'], data_fold['tpr'], 
                     lw=1, alpha=0.3, color=model_colors[best_model_name],
                     label=f'Fold {fold_idx+1}' if fold_idx == 0 else "")
        
        # 计算平均 TPR
        mean_tpr = np.mean(mean_tpr_over_fpr[i], axis=0)
        mean_tpr[-1] = 1.0
        mean_specificity = 1 - mean_fpr
        mean_auc = auc(mean_fpr, mean_tpr)
        
        # 绘制平均曲线
        plt.plot(mean_specificity, mean_tpr, 
                 color=model_colors[best_model_name],
                 lw=2.5, alpha=1.0,
                 label=f'Mean ROC (AUC = {mean_auc:.4f})')
        
        # 设置坐标轴：横坐标特异性 (1->0), 纵坐标敏感性 (0->1)
        # 为了符合视觉习惯（左上角为优），特异性轴应从 1 到 0
        plt.xlim(1, 0) 
        plt.ylim(0.0, 1.05)
        
        plt.xticks(np.arange(1, -0.1, -0.2), fontsize=10) # 1.0, 0.8, ..., 0.0
        plt.yticks(np.arange(0, 1.1, 0.2), fontsize=10)
        
        plt.xlabel('Specificity (特异性)', fontsize=12, labelpad=2)
        plt.ylabel('Sensitivity (敏感性)', fontsize=12, labelpad=2)
        plt.title(f'{plot_title} - {class_label}\n{best_model_name} 5-Fold CV ROC', fontsize=12, pad=10)
        
        plt.legend(loc='lower left', frameon=False, fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.3)
        sns.despine(trim=True)
        plt.tight_layout()
        
        plt.savefig(output_dir + f'cv_roc_{class_label}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"CV ROC for {class_label}: Mean AUC = {mean_auc:.4f}")

    # ========== 新增代码结束 ==========

    # 继续原有的测试集评估流程...
    best_model.fit(X_train_scaled, y_train)
    joblib.dump(best_model, output_dir + 'best_model.pkl')

    # 输出测试数据集的样本名称至文本文件
    test_sample_names = data.iloc[X_test.index][['Sample', 'Control_sample']]
    test_sample_names.to_csv(output_dir + 'test_sample_names.txt', sep='\t', na_rep='nan')

    # 训练集性能指标
    y_train_pred = best_model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_report = classification_report(
        y_train, y_train_pred, target_names=le.classes_)
    print(f"Best model ({best_model_name}) accuracy on training set: {train_accuracy:.4f}")
    print("Training set classification report:")
    print(train_report)

    # 测试集性能指标
    y_test_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred, target_names=le.classes_)
    print(f"Best model ({best_model_name}) accuracy on test set: {test_accuracy:.4f}")
    print("Test set classification report:")
    print(test_report)

    metrics_text = ""

    # 训练集和测试集的混淆矩阵
    datasets = {
        'train': (X_train_scaled, y_train),
        'test': (X_test_scaled, y_test)
    }

    # 注意：这里重新训练所有模型以生成混淆矩阵，保持原逻辑
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)

        for dataset_name, (X_data, y_data) in datasets.items():
            y_pred = model.predict(X_data)

            # 混淆矩阵
            cm = confusion_matrix(y_data, y_pred)

            plt.figure(figsize=(6, 4))
            custom_xticklabels = ['FR', 'IR', 'NSR']
            custom_yticklabels = ['FR', 'IR', 'NSR']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=custom_xticklabels, yticklabels=custom_yticklabels)
            plt.xlabel('Predicted label', fontsize=12)
            plt.ylabel('True label', fontsize=12)
            plt.title(plot_title, fontsize=12)
            plt.savefig(output_dir + f'{model_name}_{dataset_name}_confusion.png', bbox_inches='tight')
            plt.tight_layout()
            plt.close()

            metrics = classification_report(
                y_data, y_pred, target_names=le.classes_, output_dict=True)
            accuracy = accuracy_score(y_data, y_pred)
            metrics_text += f"\n{model_name} {dataset_name.capitalize()} Report:\n"
            metrics_text += f"Accuracy: {accuracy:.4f}\n"
            for cls in le.classes_:
                metrics_text += f"{cls} - Precision: {metrics[cls]['precision']:.4f}\n"
                metrics_text += f"{cls} - Recall: {metrics[cls]['recall']:.4f}\n"
                metrics_text += f"{cls} - F1 score: {metrics[cls]['f1-score']:.4f}\n"

            # 找出预测与真实标签不同的样本
            if dataset_name == 'train':
               indices = X_train.index
            else:
               indices = X_test.index
            misclassified_indices = np.where(y_data != y_pred)[0]
            misclassified_samples = data.iloc[indices[misclassified_indices]][['Sample', 'Control_sample']]
            misclassified_samples['True Label'] = le.inverse_transform(y_data[misclassified_indices])
            misclassified_samples['Predicted Label'] = le.inverse_transform(y_pred[misclassified_indices])

            # 保存到文本文件
            misclassified_samples.to_csv(output_dir + f'{model_name}_{dataset_name}_misclassified.txt', sep='\t',
                                        na_rep='nan')

    # ROC 曲线数据准备 (测试集)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_encoded))
    y_test_probs = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_test_probs[model_name] = model.predict_proba(X_test_scaled)

    # 保存 ROC 数据
    roc_data = []
    for class_idx in range(n_classes):
        class_label = le.classes_[class_idx]
        for model_name, probs in y_test_probs.items():
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], probs[:, class_idx])
            roc_auc = auc(fpr, tpr)
            roc_data.append({
                'model_name': model_name,
                'class_label': class_label,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            })
    with open(output_dir + 'roc_data.json', 'w') as f:
        json.dump(roc_data, f)

    # ========== 测试集多模型 ROC 绘图 (保持原有逻辑) ==========
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.0

    for class_idx in range(n_classes):
        class_label = le.classes_[class_idx]
        plt.figure(figsize=(5, 5))
        for model_name, probs in y_test_probs.items():
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], probs[:, class_idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 
                     color=model_colors[model_name],
                     lw=1.5,
                     alpha=0.8,
                     label=f'{model_name} ({roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.6)
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xticks(np.arange(0, 1.1, 0.2), fontsize=10)
        plt.yticks(np.arange(0, 1.1, 0.2), fontsize=10)
        plt.xlabel('False Positive Rate', fontsize=12, labelpad=2)
        plt.ylabel('True Positive Rate', fontsize=12, labelpad=2)
        plt.title(f'{class_label} (Test Set)', fontsize=12, pad=10)
        
        leg = plt.legend(loc='lower right', 
                        frameon=False, 
                        borderaxespad=0.2,
                        handlelength=1.5,
                        fontsize=9)
        for line in leg.get_lines():
            line.set_linewidth(2.0)
            
        sns.despine(trim=True)
        plt.tight_layout(pad=1.5)
        plt.savefig(output_dir + f'test_roc_{class_label}.png', dpi=300, bbox_inches='tight')
        plt.close()

    with open(output_dir + 'metrics.txt', 'w') as f:
        f.write(metrics_text)
