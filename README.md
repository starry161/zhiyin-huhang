# 智银护航：银行客户流失预测决策平台

这是一个可直接部署到 Streamlit Community Cloud 的银行客户流失风险分析应用。平台保留完整业务闭环：客户数据上传与清洗、XGBoost 流失预测、风险等级划分、SHAP 全局与单客户解释、风险—价值分层、运营建议以及模型效果可视化。

## 项目结构

```text
.
├── app.py
├── requirements.txt
├── .streamlit/config.toml
└── outputs/churn_model_experiment/
    ├── best_recall_model.joblib
    ├── table2_model_performance.csv
    ├── roc_curves.png
    ├── model_comparison/model_performance_comparison.csv
    └── shap_analysis/
        ├── SHAP特征重要性排序.csv
        └── shap_feature_importance.csv
```

## 本地运行

建议使用 Python 3.10 或 3.11：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开终端显示的地址即可使用。应用不依赖本机绝对路径，模型和离线分析资产均随项目一起发布。

## 输入 CSV 字段

必需字段：`CreditScore`、`Geography`、`Gender`、`Age`、`Tenure`、`Balance`、`NumOfProducts`、`HasCrCard`、`IsActiveMember`、`EstimatedSalary`。

`CustomerId` 为可选字段；缺少时平台会自动生成客户编号。`Exited` 不需要上传，平台只使用客户特征完成预测。

## 在线部署到 Streamlit Cloud

1. 将本项目完整目录上传到 GitHub（包括 `outputs/churn_model_experiment` 下的模型和 CSV/图片资产）。
2. 登录 [Streamlit Community Cloud](https://share.streamlit.io/)，点击 **New app**。
3. 选择 GitHub 仓库、分支，并将 **Main file path** 设置为 `app.py`。
4. 点击 **Deploy**。平台会自动读取 `requirements.txt` 安装依赖，并读取 `.streamlit/config.toml` 应用服务器配置。

更新代码后推送到所选分支，Streamlit Cloud 会自动重新部署。首次启动可能需要几分钟安装 SHAP 与 XGBoost 等依赖。

## 页面说明

1. **平台首页**：项目定位、核心能力与业务闭环。
2. **数据管理**：上传 CSV、字段校验、清洗检查和预览。
3. **风险识别**：批量预测、风险等级、分布图和结果下载。
4. **AI 风险解释**：全局 SHAP 重要性和单客户风险原因。
5. **AI 智能运营**：按风险与客户价值生成维护策略和运营方案。
6. **模型评估**：模型对比指标与 ROC 曲线。
7. **平台架构**：数据、模型、解释和决策链路说明。
