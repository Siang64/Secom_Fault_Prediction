# 半導體製造數據分析與故障預測系統

## 1. 專案介紹

本專案使用 SECOM 半導體製造過程監測數據集，建立一個完整的數據分析與故障預測系統。該系統通過機器學習模型分析半導體生產過程中的各種信號，以預測可能的製造故障，提高生產良率並降低成本。

本專案涵蓋了從數據探索、預處理、特徵工程到模型訓練、評估和部署的完整機器學習工作流程，適合作為工業數據分析與故障預測的案例練習。

## 2. 數據集說明

SECOM (SEmi-COnductor Manufacturing) 數據集包含:
- 1567 個樣本 (生產過程記錄)
- 590 個特徵 (各種感測器信號)
- 標籤: -1(故障), 1(正常)
- 93.36% 的樣本為故障樣本，顯示了明顯的類別不平衡

## 3. 專案結構

```
secom/
├── secom.data              # 原始特徵數據
├── secom.names             # 特徵名稱
├── secom_labels.data       # 標籤數據
├── secom_fault_prediction.ipynb   # 主要分析與模型開發的Jupyter筆記本
├── README.md               # 專案說明文件
├── .gitignore              # git忽略檔案
│
├── results/                # 預測結果
│   └── predictions.csv     # 模型預測輸出
│
├── sample_data/            # 樣本數據
│   └── test_data.csv       # 用於測試的模擬數據
│
└── semiconductor_maintenance/  # 專案主目錄
    ├── data/               # 處理後的數據
    │   ├── feature_set_anova_f.csv     # ANOVA特徵選擇後的數據
    │   ├── feature_set_mutual_info.csv # 相互資訊特徵選擇後的數據
    │   ├── feature_set_original.csv    # 原始特徵集
    │   ├── feature_set_pca.csv         # PCA降維後的數據
    │   └── preprocessed_data.csv       # 預處理後的數據
    │
    ├── models/             # 訓練完成的模型
    │   ├── gradient_boosting_model.pkl # 梯度提升模型
    │   ├── logistic_regression_model.pkl # 邏輯回歸模型
    │   ├── mlp_model.pkl               # 多層感知器模型
    │   ├── optimized_mlp_model.pkl     # 參數優化後的MLP模型
    │   ├── random_forest_model.pkl     # 隨機森林模型
    │   └── svm_model.pkl               # 支持向量機模型
    │
    ├── results/            # 分析結果
    │   ├── data_summary.txt            # 數據摘要
    │   ├── feature_statistics.csv      # 特徵統計
    │   ├── model_evaluation.csv        # 模型評估結果
    │   ├── permutation_importance.csv  # 排列重要性結果
    │   ├── selected_features.csv       # 特徵選擇結果
    │   ├── shap_importance.csv         # SHAP值分析結果
    │   └── test_performance.txt        # 測試集性能結果
    │
    └── visualizations/     # 視覺化圖表
        ├── boxplots_by_class.png       # 各類別的箱型圖
        ├── class_distribution.png      # 類別分布圖
        ├── confusion_matrix_*.png      # 各模型的混淆矩陣
        ├── correlation_heatmap.png     # 特徵相關性熱圖
        ├── feature_distributions.png   # 特徵分布圖
        ├── missing_values_heatmap.png  # 缺失值分布熱圖
        ├── pca_explained_variance.png  # PCA解釋方差圖
        ├── permutation_importance.png  # 排列重要性圖
        ├── roc_curve_*.png             # 各模型的ROC曲線
        ├── shap_dependence_*.png       # SHAP依賴圖
        ├── shap_summary_bar.png        # SHAP摘要條形圖
        └── shap_summary_dot.png        # SHAP摘要點圖
```

## 4. 專案工作流程

本專案依照以下五個階段進行開發：

### 階段 1: 數據探索與理解
- 載入與檢查數據
- 進行數據可視化探索
- 分析缺失值與數據分布特性

### 階段 2: 數據預處理
- 處理缺失值和異常值
- 進行特徵工程與選擇
- 實現ANOVA、Mutual Information、PCA等特徵選擇方法

### 階段 3: 模型開發與評估
- 建立多種基準模型(邏輯回歸、隨機森林、梯度提升、SVM、MLP)
- 進行模型評估與比較
- 對最佳模型進行參數優化

### 階段 4: 特徵重要性分析
- 使用Built-in Feature Importance
- 計算Permutation Importance
- 進行SHAP值分析

### 階段 5: 實際應用
- 生成模擬測試數據
- 使用訓練好的模型進行預測
- 提供可重用的預測函數

## 5. 主要技術

本專案使用以下技術：

- **Python**: 主要開發語言
- **Pandas & NumPy**: 數據處理與操作
- **Scikit-learn**: 機器學習模型與評估
- **Matplotlib & Seaborn**: 數據可視化
- **SHAP**: 模型解釋性分析

## 6. 專案結果與發現

- **最佳模型**: 經參數優化的多層感知器(MLP)
- **關鍵指標**: 準確率: 96.25%, 精確率: 99.63%, 召回率: 92.83%, F1分數: 96.11%
- **特徵選擇**: ANOVA F值特徵選擇方法效果最佳
- **關鍵發現**: 透過SHAP分析，識別出多個對故障預測有重要影響的關鍵特徵

## 7. 使用指南

### 環境配置

```bash
# 安裝必要套件
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib
```

### 運行預測

本系統提供兩個主要功能：生成測試數據和進行故障預測。

#### 1. 輸入數據格式

預測系統需要包含50個特徵的數據集。輸入數據範例 (`sample_data/test_data.csv`) 的前5行如下：

```
Feature_0,Feature_1,Feature_2,Feature_3,Feature_4,... (共50個特徵)
0.496714,0.138540,0.647632,0.413580,0.333674,...
0.726456,0.171434,0.918777,0.269859,0.944059,...
-0.278564,0.255475,0.600139,-0.908055,0.738410,...
0.513587,-0.998634,0.267314,0.968603,-0.235619,...
...
```

#### 2. 使用Python API進行預測

```python
# 載入並使用預測函數
from test import predict_failures

# 使用自定義數據進行預測
import pandas as pd
input_data = pd.read_csv('your_data.csv')
results = predict_failures(input_data=input_data)

# 或使用模擬測試數據
from test import generate_test_data
test_data = generate_test_data()
results = predict_failures(input_data=test_data)
```

#### 3. 輸出結果範例

預測結果將保存至 `results/predictions.csv`，格式如下：

```
原始索引,預測結果,故障機率
0,故障,0.9856542563438416
1,正常,0.07075681537389755
2,故障,0.9483073353767395
3,故障,0.9966093897819519
4,正常,0.04695932194590569
...
```

輸出說明：
- **原始索引**：對應輸入數據的行索引
- **預測結果**：「故障」或「正常」的預測結果
- **故障機率**：模型預測為故障的概率值（0-1之間）

預測結果範例顯示：

| 原始索引 | 預測結果 | 故障機率 |
|---------|---------|---------|
| 0       | 故障     | 0.985654|
| 1       | 正常     | 0.070757|
| 2       | 故障     | 0.948307|
| 3       | 故障     | 0.996609|
| 4       | 正常     | 0.046959|
| ...     | ...     | ...     |


## 8. 參考資源

- [SECOM數據集介紹](https://archive.ics.uci.edu/ml/datasets/SECOM)
