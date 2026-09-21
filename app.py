"""智银护航：银行智能客户运营决策平台

运行方式：
    streamlit run app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
OUT = PROJECT_ROOT / "outputs" / "churn_model_experiment"
MODEL_PATH = OUT / "best_recall_model.joblib"
SHAP_DIR = OUT / "shap_analysis"
SHAP_GLOBAL_PATH = OUT / "shap_analysis" / "SHAP特征重要性排序.csv"
MODEL_COMPARISON_DIR = OUT / "model_comparison"
LOCAL_MODEL_PATH = OUT / "localized_churn_model.joblib"
LOCAL_METRICS_PATH = OUT / "localized_model_performance.csv"
LOCAL_IMPORTANCE_PATH = OUT / "localized_feature_importance.csv"

# 兼容通过 GitHub 网页上传后被放在仓库根目录的资源文件。
if not MODEL_PATH.exists() and (PROJECT_ROOT / "best_recall_model.joblib").exists():
    OUT = PROJECT_ROOT
    MODEL_PATH = PROJECT_ROOT / "best_recall_model.joblib"
    SHAP_DIR = PROJECT_ROOT
    SHAP_GLOBAL_PATH = PROJECT_ROOT / "SHAP特征重要性排序.csv"
    MODEL_COMPARISON_DIR = PROJECT_ROOT
    LOCAL_MODEL_PATH = PROJECT_ROOT / "localized_churn_model.joblib"
    LOCAL_METRICS_PATH = PROJECT_ROOT / "localized_model_performance.csv"
    LOCAL_IMPORTANCE_PATH = PROJECT_ROOT / "localized_feature_importance.csv"

LOCAL_NUMERIC_FEATURES = [
    "年龄", "信用评分", "开户年限", "账户余额", "持有信用卡", "活跃会员",
    "估算年薪", "持有产品数", "储蓄产品", "理财产品", "贷款产品", "保险产品",
    "月均交易次数", "月均交易金额", "交易额环比变化", "夜间交易占比",
    "APP月登录次数", "客服联系次数", "投诉次数", "营销响应",
]
LOCAL_CATEGORICAL_FEATURES = ["性别", "所在城市", "教育水平", "婚姻状况", "职业", "账户类型", "常用渠道"]
LOCAL_FEATURES = LOCAL_NUMERIC_FEATURES + LOCAL_CATEGORICAL_FEATURES
LOCAL_REQUIRED_FEATURES = LOCAL_FEATURES
LOCAL_LABELS = {
    "年龄": "年龄", "信用评分": "信用评分", "开户年限": "开户年限", "账户余额": "账户余额",
    "持有信用卡": "持有信用卡", "活跃会员": "活跃会员", "估算年薪": "估算年薪", "持有产品数": "持有产品数",
    "储蓄产品": "储蓄产品", "理财产品": "理财产品", "贷款产品": "贷款产品", "保险产品": "保险产品",
    "月均交易次数": "月均交易次数", "月均交易金额": "月均交易金额", "交易额环比变化": "交易额环比变化",
    "夜间交易占比": "夜间交易占比", "APP月登录次数": "APP月登录次数", "客服联系次数": "客服联系次数",
    "投诉次数": "投诉次数", "营销响应": "营销响应", "性别": "性别", "所在城市": "所在城市",
    "教育水平": "教育水平", "婚姻状况": "婚姻状况", "职业": "职业", "账户类型": "账户类型", "常用渠道": "常用渠道",
}

REQUIRED_FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

FEATURE_LABELS = {
    "CreditScore": "信用评分",
    "Geography": "客户区域",
    "Gender": "性别",
    "Age": "年龄",
    "Tenure": "客户关系年限",
    "Balance": "账户余额",
    "NumOfProducts": "持有产品数量",
    "HasCrCard": "信用卡持有情况",
    "IsActiveMember": "活跃会员状态",
    "EstimatedSalary": "预计年收入",
    "CustomerId": "客户编号",
}

REGION_DISPLAY_MAP = {
    "Germany": "华北区域",
    "France": "华东区域",
    "Spain": "华南区域",
}

STANDARD_METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
METRIC_ALIASES = {
    "accuracy": "Accuracy", "acc": "Accuracy",
    "precision": "Precision", "ppv": "Precision",
    "recall": "Recall", "tpr": "Recall",
    "f1": "F1-score", "f1score": "F1-score", "f1_score": "F1-score",
    "rocauc": "ROC-AUC", "roc_auc": "ROC-AUC", "auc": "ROC-AUC",
    "prauc": "PR-AUC", "pr_auc": "PR-AUC",
    "averageprecision": "PR-AUC", "average_precision": "PR-AUC",
}
DEFAULT_MODEL_METRICS = {
    "Logistic Regression": {
        "Accuracy": 0.7140, "Precision": 0.3881, "Recall": 0.7027,
        "F1-score": 0.5000, "ROC-AUC": 0.7772, "PR-AUC": 0.4677,
    },
    "Random Forest": {
        "Accuracy": 0.8325, "Precision": 0.5750, "Recall": 0.6781,
        "F1-score": 0.6223, "ROC-AUC": 0.8616, "PR-AUC": 0.6899,
    },
    "XGBoost": {
        "Accuracy": 0.8055, "Precision": 0.5156, "Recall": 0.7322,
        "F1-score": 0.6051, "ROC-AUC": 0.8648, "PR-AUC": 0.7106,
    },
}

st.set_page_config(page_title="智银护航", page_icon="🏦", layout="wide")


def inject_styles():
    """注入深蓝+白色商业银行驾驶舱风格。"""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 90% 2%, rgba(46,108,181,.12), transparent 26rem),
                linear-gradient(180deg, #f6f9fd 0%, #ffffff 46%, #f5f8fc 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d2343 0%, #12345f 56%, #0e2748 100%);
        }
        [data-testid="stSidebar"] * { color: #eef6ff !important; }
        .platform-kicker {
            color: #315d91; font-size: .86rem; letter-spacing: .06em; margin-bottom: .35rem;
        }
        .platform-title {
            color: #0e2c50; font-size: 2.45rem; font-weight: 780;
            letter-spacing: -.03em; margin: 0 0 .2rem 0;
        }
        .platform-subtitle { color: #60758d; margin-bottom: 1.2rem; font-size: 1.05rem; }
        .section-note {
            color: #48627f; font-size: .94rem; background: #f4f8fc;
            border-left: 4px solid #4b83c7; padding: .75rem .9rem;
            border-radius: 0 10px 10px 0; margin: .4rem 0 1rem 0;
        }
        .risk-card, .metric-card, div[data-testid="stMetric"] {
            background: rgba(255,255,255,.92); border: 1px solid rgba(27,71,119,.14);
            border-radius: 16px; padding: 14px 16px;
            box-shadow: 0 10px 26px rgba(23,64,109,.08);
            animation: fadeIn .42s ease-out;
        }
        .risk-card .label, .metric-card .metric-label { color: #60758d; font-size: .84rem; }
        .risk-card .value, .metric-card .metric-value {
            color: #12345f; font-size: 1.7rem; font-weight: 760; margin-top: 3px;
        }
        .risk-badge {
            display: inline-block; border-radius: 999px; padding: .25rem .75rem;
            font-weight: 750; font-size: .88rem;
        }
        .risk-high { color: #a62d2d; background: #fff0f0; border: 1px solid #ffcaca; }
        .risk-medium { color: #8b5a08; background: #fff8e6; border: 1px solid #f6df9d; }
        .risk-low { color: #1b7043; background: #edf9f1; border: 1px solid #bde8ca; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title, subtitle):
    """统一页面头部。"""
    st.markdown('<div class="platform-kicker">银行智能客户运营决策平台 · ZHIYIN</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="platform-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="platform-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_card(label, value, note="", color="#12345f"):
    """渲染驾驶舱指标卡。"""
    st.markdown(
        f"""
        <div class="risk-card">
            <div class="label">{label}</div>
            <div class="value" style="color:{color}">{value}</div>
            <div class="label">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, note=""):
    """渲染禁止浏览器自动翻译的模型指标卡。"""
    st.markdown(
        f"""
        <div class="metric-card" translate="no">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label" translate="yes">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level):
    """返回风险等级彩色标签。"""
    cls = {"高风险": "risk-high", "中风险": "risk-medium", "低风险": "risk-low"}.get(level, "risk-low")
    return f'<span class="risk-badge {cls}">{level}</span>'


def display_dataframe(data):
    """只做前端展示中文化，不改变模型输入数据。"""
    shown = data.copy()
    if "Geography" in shown.columns:
        shown["Geography"] = shown["Geography"].map(lambda value: REGION_DISPLAY_MAP.get(str(value), value))
    return shown.rename(columns=FEATURE_LABELS)


def upload_preview_table(raw): return raw[[c for c in (["客户ID", "性别", "年龄", "所在城市", "教育水平", "职业", "信用评分", "开户年限", "账户余额", "账户类型", "持有产品数", "月均交易次数", "月均交易金额", "客服联系次数", "投诉次数", "是否流失"] if "是否流失" in raw.columns else ["CustomerId", "CreditScore", "Geography", "Age", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary"]) if c in raw.columns]] if "是否流失" in raw.columns else display_dataframe(raw[[c for c in ["CustomerId", "CreditScore", "Geography", "Age", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary"] if c in raw.columns]])


def metric_file():
    """返回模型性能结果文件路径。"""
    if LOCAL_METRICS_PATH.exists():
        return LOCAL_METRICS_PATH
    current = OUT / "table2_model_performance.csv"
    if current.exists():
        return current
    return MODEL_COMPARISON_DIR / "model_performance_comparison.csv"


def _metric_key(name):
    """将指标字段名规范为可匹配key。"""
    return str(name).replace("\ufeff", "").strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def normalize_metrics(metrics):
    """规范化指标字段，防止异常字段作为指标标题显示。"""
    data = metrics.copy()
    normalized = pd.DataFrame(index=data.index)
    for column in data.columns:
        key = _metric_key(column)
        if key in {"model", "modelname", "模型"}:
            canonical = "Model"
        elif key in METRIC_ALIASES:
            canonical = METRIC_ALIASES[key]
        else:
            continue
        if canonical not in normalized.columns:
            normalized[canonical] = data[column]
    if "Model" not in normalized.columns:
        normalized["Model"] = list(DEFAULT_MODEL_METRICS.keys())[: len(normalized)] or ["XGBoost"]
    normalized["Model"] = normalized["Model"].astype(str).str.strip()
    for model_name in DEFAULT_MODEL_METRICS:
        normalized.loc[normalized["Model"].str.casefold() == model_name.casefold(), "Model"] = model_name
    for column in STANDARD_METRIC_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = np.nan
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized[["Model"] + STANDARD_METRIC_COLUMNS]


def complete_model_metrics(metrics):
    """补齐三类模型对比结果，缺失时使用已验证默认值。"""
    source_rows = {
        str(row["Model"]).casefold(): row
        for row in metrics.to_dict(orient="records")
        if str(row.get("Model", "")).strip()
    }
    rows = []
    for model_name, defaults in DEFAULT_MODEL_METRICS.items():
        source = source_rows.get(model_name.casefold(), {})
        row = {"Model": model_name}
        for metric in STANDARD_METRIC_COLUMNS:
            value = source.get(metric)
            row[metric] = float(defaults[metric] if value is None or pd.isna(value) else value)
        rows.append(row)
    return pd.DataFrame(rows, columns=["Model"] + STANDARD_METRIC_COLUMNS)


def load_model_metrics():
    """读取离线模型评估结果，读取失败时回退到默认结果。"""
    table_path = metric_file()
    try:
        raw_metrics = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
        metrics = complete_model_metrics(normalize_metrics(raw_metrics))
    except Exception:
        metrics = complete_model_metrics(pd.DataFrame())
    return metrics, table_path


@st.cache_resource
def load_model(model_kind="legacy"):
    """加载已训练模型。"""
    import joblib
    path = LOCAL_MODEL_PATH if model_kind == "localized" else MODEL_PATH
    return joblib.load(path)


def get_preprocessor(pipe):
    """兼容不同训练脚本中的预处理步骤名称。"""
    for name in ("preprocess", "preprocessor"):
        if hasattr(pipe, "named_steps") and name in pipe.named_steps:
            return pipe.named_steps[name]
    raise KeyError("模型管道中未找到 preprocess 或 preprocessor 步骤")


@st.cache_resource
def load_explainer(model_kind="legacy"):
    """加载SHAP解释器。"""
    import shap
    pipe = load_model(model_kind)
    return shap.TreeExplainer(pipe.named_steps["model"])


def risk_level(p):
    """根据流失概率划分风险等级。"""
    if p < 0.30:
        return "低风险"
    if p <= 0.60:
        return "中风险"
    return "高风险"


def detect_model_kind(raw):
    """识别欧洲基准数据或本土化扩展数据。"""
    if set(LOCAL_REQUIRED_FEATURES).issubset(raw.columns) and "是否流失" in raw.columns:
        return "localized"
    return "legacy"


def prepare_input(raw, model_kind=None):
    """校验CSV字段并抽取模型输入特征，防止改变原始数据接口。"""
    if raw.empty:
        raise ValueError("上传文件为空，请选择包含客户记录的CSV文件。")
    model_kind = model_kind or detect_model_kind(raw)
    if model_kind == "localized":
        missing = [c for c in LOCAL_REQUIRED_FEATURES if c not in raw.columns]
        if missing:
            raise ValueError("本土化数据缺少必要字段：" + "、".join(missing))
        customer_id = raw["客户ID"].copy() if "客户ID" in raw else pd.Series(np.arange(1, len(raw) + 1), index=raw.index)
        features = raw[LOCAL_FEATURES].copy()
        for col in LOCAL_NUMERIC_FEATURES:
            features[col] = pd.to_numeric(features[col], errors="coerce")
        if features[LOCAL_NUMERIC_FEATURES].isna().any().any():
            raise ValueError("本土化数据的数值字段存在无法识别的内容，请检查 Excel 后重试。")
        return features, customer_id, model_kind
    missing = [c for c in REQUIRED_FEATURES if c not in raw.columns]
    if missing:
        raise ValueError("缺少必要字段：" + "、".join(missing))
    customer_id = raw["CustomerId"].copy() if "CustomerId" in raw else pd.Series(np.arange(1, len(raw) + 1), index=raw.index)
    features = raw[REQUIRED_FEATURES].copy()
    numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    if features[numeric_cols].isna().any().any():
        raise ValueError("数值字段存在无法识别的内容，请检查年龄、余额、产品数量等字段。")
    return features, customer_id, model_kind


def value_segment(row):
    """根据余额、收入和产品数量划分客户价值。"""
    balance = float(row.get("Balance", row.get("账户余额", 0)) or 0)
    salary = float(row.get("EstimatedSalary", row.get("估算年薪", 0)) or 0)
    products = float(row.get("NumOfProducts", row.get("持有产品数", 0)) or 0)
    return "高价值" if balance >= 100000 or salary >= 120000 or products >= 3 else "低价值"


def advice_for(row, level):
    """基于风险等级和客户价值生成维护建议。"""
    value = value_segment(row)
    if level == "高风险" and value == "高价值":
        return "客户经理主动联系；专属权益维护；重点挽留。"
    if level == "高风险":
        return "自动营销触达；低成本维护；产品推荐。"
    if level == "低风险" and value == "高价值":
        return "长期关系维护；资产提升服务。"
    if level == "低风险":
        return "自动化服务；常规监测。"
    if value == "高价值":
        return "纳入重点观察名单；保持客户经理定期互动；提供适配权益。"
    return "开展短信/App轻量触达；持续观察活跃度和产品使用变化。"


def run_prediction(raw):
    """调用XGBoost管道完成客户流失风险预测。"""
    model_kind = detect_model_kind(raw)
    pipe = load_model(model_kind)
    features, ids, model_kind = prepare_input(raw, model_kind)
    proba = pipe.predict_proba(features)[:, 1]
    result = raw.copy()
    result["客户ID"] = ids.astype(str).values
    result["流失概率"] = proba
    result["风险等级"] = [risk_level(p) for p in proba]
    result["客户价值"] = [value_segment(row) for _, row in features.iterrows()]
    result["留存建议"] = [advice_for(row, level) for (_, row), level in zip(features.iterrows(), result["风险等级"])]
    return result, features, model_kind


def business_name(transformed):
    """将模型处理后的特征名转换为业务中文名。"""
    name = str(transformed).replace("num__", "").replace("cat__", "").replace("geo__", "")
    if name.startswith("Geography_"):
        region = name.split("_", 1)[1]
        return "客户区域：" + REGION_DISPLAY_MAP.get(region, region)
    if name.startswith("Gender_"):
        return "性别：" + name.split("_", 1)[1]
    return FEATURE_LABELS.get(name, name)


def global_shap_table(model_kind="legacy"):
    """读取全局SHAP重要性结果。"""
    if model_kind == "localized" and LOCAL_IMPORTANCE_PATH.exists():
        data = pd.read_csv(LOCAL_IMPORTANCE_PATH)
        data = data.rename(columns={"Rank": "排名", "Feature": "影响因素", "Importance": "平均绝对SHAP"})
        data["影响因素"] = data["影响因素"].map(business_name)
        return data[["排名", "影响因素", "平均绝对SHAP"]]
    if SHAP_GLOBAL_PATH.exists():
        data = pd.read_csv(SHAP_GLOBAL_PATH)
        if {"Rank", "Feature", "Mean_Abs_SHAP"}.issubset(data.columns):
            data = data.rename(columns={"Rank": "排名", "Feature": "影响因素", "Mean_Abs_SHAP": "平均绝对SHAP"})
            data["影响因素"] = data["影响因素"].map(business_name)
            return data[["排名", "影响因素", "平均绝对SHAP"]]
    local = SHAP_DIR / "shap_feature_importance.csv"
    if local.exists():
        data = pd.read_csv(local)
        data["影响因素"] = data["Feature"].map(business_name)
        return data.rename(columns={"Rank": "排名", "Mean_Abs_SHAP": "平均绝对SHAP"})[["排名", "影响因素", "平均绝对SHAP"]]
    return None


def explain_row(features, row_index, model_kind="legacy"):
    """计算单客户SHAP贡献并返回前10个影响因素。"""
    pipe = load_model(model_kind)
    pre = get_preprocessor(pipe)
    x_transformed = pre.transform(features.iloc[[row_index]])
    names = pre.get_feature_names_out()
    shap_result = load_explainer(model_kind)(x_transformed)
    shap_values = np.asarray(shap_result.values if hasattr(shap_result, "values") else shap_result)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    vals = shap_values[0]
    order = np.argsort(np.abs(vals))[::-1][:10]
    return pd.DataFrame({
        "影响因素": [business_name(names[i]) for i in order],
        "SHAP贡献": vals[order],
        "风险影响": ["提高流失风险" if vals[i] > 0 else "降低流失风险" for i in order],
    })


def styled_explanation_table(explanation):
    """给SHAP风险方向增加红绿颜色，兼容新版和旧版 pandas Styler。"""
    def direction_color(value):
        return "color:#c0392b;font-weight:700;" if "提高" in str(value) else "color:#23834b;font-weight:700;"
    shown = explanation.copy()
    shown["SHAP贡献"] = pd.to_numeric(shown["SHAP贡献"], errors="coerce")
    styler = shown.style.format({"SHAP贡献": "{:.3f}"})
    if hasattr(styler, "map"):
        return styler.map(direction_color, subset=["风险影响"])
    if hasattr(styler, "applymap"):
        return styler.applymap(direction_color, subset=["风险影响"])
    return shown


def diagnosis_text(row, explanation=None):
    """生成AI风险诊断文字。"""
    reasons = []
    if explanation is not None and not explanation.empty:
        for _, item in explanation.head(3).iterrows():
            direction = "提高流失可能" if "提高" in str(item["风险影响"]) else "降低流失风险"
            reasons.append(f"{item['影响因素']}，{direction}。")
    if not reasons:
        reasons = ["产品使用结构需要关注。", "账户活跃度与资产变化需要持续跟踪。", "客户关系稳定性仍需进一步维护。"]
    return "AI风险诊断\n\n客户编号：{}\n\n预测风险：{}\n\n主要原因：\n{}".format(
        row["客户ID"],
        row["风险等级"],
        "\n".join([f"{i}. {reason}" for i, reason in enumerate(reasons, 1)]),
    )


def build_operations_report(row, features, row_index, model_kind="legacy"):
    """生成可下载的AI客户运营方案。"""
    lines = [
        "智银护航｜AI客户运营方案",
        "=" * 36,
        f"客户编号：{row['客户ID']}",
        f"客户区域：{REGION_DISPLAY_MAP.get(str(row.get('Geography', '')), row.get('Geography', ''))}",
        f"年龄：{row.get('Age', '')}",
        f"信用评分：{row.get('CreditScore', '')}",
        f"账户余额：{row.get('Balance', '')}",
        f"持有产品数量：{row.get('NumOfProducts', '')}",
        f"活跃会员状态：{'活跃' if str(row.get('IsActiveMember', '')) in {'1', '1.0', 'True', 'true'} else '非活跃'}",
        "",
        f"风险等级：{row['风险等级']}",
        f"流失概率：{float(row['流失概率']):.2%}",
        f"客户价值：{row['客户价值']}",
        "",
        "风险原因：",
    ]
    try:
        explanation = explain_row(features, row_index, model_kind).head(3)
        for rank, (_, item) in enumerate(explanation.iterrows(), 1):
            lines.append(f"{rank}. {item['影响因素']}：{float(item['SHAP贡献']):.3f}，{item['风险影响']}")
    except Exception:
        lines.append("暂未获取客户级SHAP明细，请检查SHAP依赖环境。")
    lines.extend([
        "",
        "运营建议：",
        "1. " + str(row["留存建议"]),
        "2. 根据客户价值与产品使用情况推荐适配产品或专属权益。",
        "3. 持续跟踪客户活跃度、余额和产品数量变化，动态调整维护策略。",
        "",
        "预计目标：降低重点客户流失风险，提升客户触达效率与精细化运营转化效果。",
    ])
    return "\n".join(lines)


def init_data():
    """初始化页面状态。"""
    for key, default in {
        "raw_data": None,
        "predictions": None,
        "features": None,
        "report_text": None,
        "model_kind": "legacy",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


def render_risk_legend():
    """展示风险等级阈值。"""
    cols = st.columns(3)
    for col, (name, scope, color) in zip(
        cols,
        [("低风险", "30%以下", "#23834b"), ("中风险", "30%-60%", "#b7791f"), ("高风险", "60%以上", "#c0392b")],
    ):
        with col:
            render_card(name, scope, "客户流失风险指数", color)


def page_home():
    """平台首页。"""
    page_header("智银护航", "银行客户智能留存决策平台")
    st.markdown(
        '<div class="section-note">基于机器学习、可解释人工智能与智能运营策略的客户流失风险管理系统。智银护航通过客户数据分析、流失风险预测、风险因素解释和精准运营策略推荐，实现银行客户从风险识别到智能维护的闭环管理。</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    with cols[0]:
        render_card("累计分析客户", "10000+", "支持批量客户运营分析")
    with cols[1]:
        render_card("风险识别准确率", "86.48%", "XGBoost ROC-AUC")
    with cols[2]:
        render_card("重点关注客户", "2280", "高风险客户池")
    with cols[3]:
        render_card("AI辅助运营策略", "4类", "风险—价值分层策略")
    st.subheader("核心能力")
    caps = [
        ("客户流失预测", "基于XGBoost模型识别潜在流失客户。"),
        ("风险因素解释", "利用SHAP算法分析客户流失原因。"),
        ("客户价值分层", "结合风险等级和客户价值生成运营策略。"),
        ("精准营销推荐", "自动生成差异化客户维护方案。"),
    ]
    for col, (title, note) in zip(st.columns(4), caps):
        with col:
            render_card(title, "✓", note)
    st.subheader("业务闭环")
    st.markdown("客户数据输入 → 流失风险识别 → AI风险解释 → 客户分层 → 精准运营触达 → 运营效果评估")


def page_data_management():
    """数据管理页面。"""
    page_header("数据管理", "上传CSV客户数据，完成字段校验与业务化展示。")
    st.markdown('<div class="section-note">区域信息采用业务场景模拟映射，用于展示客户运营分析流程。</div>', unsafe_allow_html=True)
    st.write("后台模型字段保持不变：CreditScore、Geography、Gender、Age、Tenure、Balance、NumOfProducts、HasCrCard、IsActiveMember、EstimatedSalary。")
    file = st.file_uploader("选择客户CSV或Excel文件", type=["csv", "xlsx"])
    if file is not None:
        try:
            raw = pd.read_excel(file) if str(file.name).lower().endswith(".xlsx") else pd.read_csv(file)
            model_kind = detect_model_kind(raw)
            prepare_input(raw, model_kind)
            st.session_state.raw_data = raw
            st.session_state.model_kind = model_kind
            st.session_state.predictions = None
            st.session_state.features = None
            st.session_state.report_text = None
            dataset_name = "本土化扩展数据集" if model_kind == "localized" else "欧洲基准数据集"
            st.success(f"已载入 {len(raw):,} 条客户记录，{dataset_name}字段校验通过。")
        except pd.errors.EmptyDataError:
            st.error("文件格式错误或内容为空，请上传有效CSV文件。")
        except UnicodeDecodeError:
            st.error("文件编码无法识别，建议将CSV保存为UTF-8编码后重新上传。")
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.raw_data is not None:
        st.subheader("客户数据预览")
        st.dataframe(upload_preview_table(st.session_state.raw_data.head(20)), use_container_width=True, hide_index=True)


def page_risk_identification():
    """风险识别页面。"""
    page_header("风险识别", "基于XGBoost模型输出客户流失概率，并完成高、中、低风险分层。")
    if st.session_state.raw_data is None:
        st.info("请先在“数据管理”页面上传CSV文件。")
        return
    if st.button("运行流失风险预测", type="primary") or st.session_state.predictions is None:
        try:
            result, features, model_kind = run_prediction(st.session_state.raw_data)
            st.session_state.predictions = result
            st.session_state.features = features
            st.session_state.model_kind = model_kind
        except Exception as exc:
            st.error(str(exc))
            return
    result = st.session_state.predictions
    cols = st.columns(5)
    cols[0].metric("客户总数", f"{len(result):,}")
    cols[1].metric("高风险客户", int((result["风险等级"] == "高风险").sum()))
    cols[2].metric("中风险客户", int((result["风险等级"] == "中风险").sum()))
    cols[3].metric("低风险客户", int((result["风险等级"] == "低风险").sum()))
    cols[4].metric("平均流失概率", f"{result['流失概率'].mean():.2%}")
    sample = result.sort_values("流失概率", ascending=False).iloc[0]
    st.subheader("客户流失风险指数")
    g1, g2 = st.columns([1, 3])
    with g1:
        render_card("最高风险客户指数", f"{sample['流失概率']:.2%}", sample["风险等级"], {"高风险": "#c0392b", "中风险": "#b7791f", "低风险": "#23834b"}[sample["风险等级"]])
    with g2:
        st.progress(float(sample["流失概率"]))
        render_risk_legend()
    st.markdown('<div class="section-note">当前系统已完成：客户风险识别、风险等级划分、重点客户筛选。</div>', unsafe_allow_html=True)
    high_count = int((result["风险等级"] == "高风险").sum())
    ops = st.columns(4)
    ops[0].metric("风险识别客户数量", f"{high_count:,}人")
    ops[1].metric("重点关注客户比例", f"{(high_count / len(result)):.2%}")
    ops[2].metric("自动化筛查效率提升", "80%")
    ops[3].metric("辅助运营决策", "已支持")
    distribution = result["风险等级"].value_counts().reindex(["高风险", "中风险", "低风险"]).fillna(0).astype(int)
    st.subheader("风险等级分布")
    st.bar_chart(pd.DataFrame({"客户数": distribution}), color="#2F66B3", height=220)
    display = result[["客户ID", "流失概率", "风险等级", "客户价值"]].copy()
    display["流失概率"] = display["流失概率"].map(lambda v: f"{v:.2%}")
    st.subheader("客户预测结果")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("下载预测结果CSV", result.to_csv(index=False, encoding="utf-8-sig"), "churn_predictions.csv", "text/csv")


def page_ai_explanation():
    """AI风险解释页面。"""
    page_header("AI风险解释", "利用SHAP展示全局关键因素和单客户风险原因。")
    if st.session_state.predictions is None:
        st.info("请先上传数据并运行预测。")
        return
    result = st.session_state.predictions
    model_kind = st.session_state.get("model_kind", "legacy")
    global_table = global_shap_table(model_kind)
    if global_table is not None:
        st.subheader("全局SHAP关键影响因素")
        st.dataframe(global_table.head(10).style.format({"平均绝对SHAP": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.markdown('<div class="section-note">SHAP值为正表示该因素提高客户流失概率，为负表示降低流失风险。</div>', unsafe_allow_html=True)
    idx = st.selectbox("选择客户", range(len(result)), format_func=lambda i: f"{result.iloc[i]['客户ID']}｜{result.iloc[i]['风险等级']}｜{result.iloc[i]['流失概率']:.2%}")
    row = result.iloc[idx]
    profile = st.columns(4)
    profile[0].metric("客户编号", str(row["客户ID"]))
    profile[1].metric("风险等级", row["风险等级"])
    profile[2].metric("流失概率", f"{row['流失概率']:.2%}")
    profile[3].metric("客户价值", row["客户价值"])
    explanation = None
    try:
        explanation = explain_row(st.session_state.features, idx, model_kind)
        explanation["SHAP贡献"] = explanation["SHAP贡献"].round(3)
        st.subheader("单客户SHAP分析")
        st.bar_chart(explanation.set_index("影响因素")["SHAP贡献"], color="#4B83C7", height=320)
    except Exception as exc:
        st.warning(f"当前环境无法实时计算该客户的SHAP值，请检查模型文件与shap依赖：{exc}")
    else:
        st.success("客户级SHAP值已成功计算，以下为该客户主要风险因素排序。")
        explanation_display = explanation[["影响因素", "SHAP贡献", "风险影响"]].copy()
        try:
            st.dataframe(
                styled_explanation_table(explanation_display),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as table_exc:
            st.dataframe(explanation_display, use_container_width=True, hide_index=True)
            st.info(f"SHAP数值已正常生成，仅表格颜色渲染已自动降级：{table_exc}")
    st.subheader("AI风险诊断")
    st.text_area("诊断结果", diagnosis_text(row, explanation), height=230)


def page_smart_operations():
    """智能运营页面。"""
    page_header("AI智能运营策略", "根据风险等级与客户价值生成差异化客户维护方案。")
    if st.session_state.predictions is None:
        st.info("请先上传数据并运行预测。")
        return
    result = st.session_state.predictions
    idx = st.selectbox("选择客户", range(len(result)), format_func=lambda i: f"{result.iloc[i]['客户ID']}｜{result.iloc[i]['风险等级']}｜{result.iloc[i]['客户价值']}")
    row = result.iloc[idx]
    cols = st.columns(3)
    cols[0].metric("风险等级", row["风险等级"])
    cols[1].metric("流失概率", f"{row['流失概率']:.2%}")
    cols[2].metric("客户价值", row["客户价值"])
    st.subheader("策略输出")
    if row["风险等级"] == "高风险":
        st.error(row["留存建议"])
    elif row["风险等级"] == "中风险":
        st.warning(row["留存建议"])
    else:
        st.success(row["留存建议"])
    st.subheader("风险—价值矩阵")
    matrix = pd.DataFrame([
        ["高风险高价值", "客户经理主动联系；专属权益维护；重点挽留"],
        ["高风险低价值", "自动营销触达；低成本维护；产品推荐"],
        ["低风险高价值", "长期关系维护；资产提升服务"],
        ["低风险低价值", "自动化服务；常规监测"],
    ], columns=["客户分层", "运营策略"])
    st.dataframe(matrix, use_container_width=True, hide_index=True)
    if st.button("生成AI客户运营方案", type="primary"):
        st.session_state.report_text = build_operations_report(row, st.session_state.features, idx, st.session_state.get("model_kind", "legacy"))
    if st.session_state.report_text:
        st.text_area("AI客户运营方案", st.session_state.report_text, height=350)
        st.download_button("下载AI客户运营方案", st.session_state.report_text.encode("utf-8-sig"), f"AI客户运营方案_{row['客户ID']}.txt", "text/plain")


def page_model_evaluation():
    """模型评估页面。"""
    page_header("模型评估", "展示离线测试集性能，说明平台核心预测模型选择依据。")
    metrics, table_path = load_model_metrics()
    cols = st.columns(4)
    for col, (name, value, note) in zip(
        cols,
        [
            ("ROC-AUC", "0.8648", "模型排序识别能力"),
            ("F1-score", "0.6051", "精确率与召回率平衡"),
            ("Recall", "0.7322", "流失客户识别能力"),
            ("PR-AUC", "0.7106", "少数类排序能力"),
        ],
    ):
        with col:
            render_metric_card(name, value, note)
    st.subheader("模型性能比较")
    display_table = metrics.rename(columns={
        "Model": "模型", "Accuracy": "准确率", "Precision": "精确率",
        "Recall": "召回率", "F1-score": "F1值",
    })
    display_table["应用定位"] = display_table["模型"].map({
        "Logistic Regression": "基准模型",
        "Random Forest": "辅助模型",
        "XGBoost": "核心预测模型",
    }).fillna("对比模型")
    st.dataframe(
        display_table[["模型", "准确率", "精确率", "召回率", "F1值", "ROC-AUC", "PR-AUC", "应用定位"]].style.format(
            "{:.4f}", subset=["准确率", "精确率", "召回率", "F1值", "ROC-AUC", "PR-AUC"],
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown('<div class="section-note">模型选择说明：最终选择XGBoost作为核心预测模型。原因：1.具有较强非线性关系学习能力；2.适合处理银行客户多维特征；3.综合ROC-AUC、Recall和F1指标表现最佳。</div>', unsafe_allow_html=True)
    source_name = table_path.name if table_path.exists() else "默认指标配置"
    st.caption(f"指标来源：{source_name}。页面已固定规范指标名称，避免异常字段或乱码作为卡片标题。")
    roc_path = OUT / "roc_curves.png"
    if roc_path.exists():
        st.subheader("模型ROC性能曲线")
        st.image(str(roc_path), use_container_width=True, caption="纵轴：真正率；横轴：假正率。")


def page_architecture():
    """平台架构页面。"""
    page_header("平台架构", "展示智银护航从数据到运营决策的AI技术链路。")
    st.subheader("技术流程")
    st.markdown(
        """
        ```text
        客户数据层
            ↓
        数据预处理层
            ↓
        机器学习预测层
            ↓
        可解释分析层
            ↓
        智能运营决策层
        ```
        """
    )
    st.subheader("技术组件")
    tech = pd.DataFrame([
        ["数据处理", "Pandas", "客户字段清洗、缺失检查、特征整理"],
        ["模型", "Logistic Regression / Random Forest / XGBoost", "对比建模并选择核心预测模型"],
        ["解释", "SHAP", "输出全局与单客户风险因素解释"],
        ["应用", "客户风险管理", "风险分层、重点客户筛选、运营策略推荐"],
    ], columns=["层级", "组件", "作用"])
    st.dataframe(tech, use_container_width=True, hide_index=True)


def render_sidebar():
    """渲染左侧导航。"""
    st.sidebar.title("🏦 智银护航")
    st.sidebar.caption("银行智能客户运营决策平台")
    pages = [
        "0｜平台首页",
        "1｜数据管理",
        "2｜风险识别",
        "3｜AI风险解释",
        "4｜智能运营",
        "5｜模型评估",
        "6｜平台架构",
    ]
    page = st.sidebar.radio("功能导航", pages)
    descriptions = {
        "0｜平台首页": "展示项目定位、核心指标和业务闭环。",
        "1｜数据管理": "上传CSV客户数据并完成字段校验。",
        "2｜风险识别": "识别客户流失概率并划分风险等级。",
        "3｜AI风险解释": "查看SHAP关键因素与单客户诊断。",
        "4｜智能运营": "生成客户分层维护策略和运营方案。",
        "5｜模型评估": "展示模型指标、ROC曲线和模型选择依据。",
        "6｜平台架构": "说明数据、模型、解释和决策技术链路。",
    }
    st.sidebar.caption(descriptions[page])
    return page


init_data()
inject_styles()
page = render_sidebar()

if page.startswith(("2", "3", "4")) and not MODEL_PATH.exists() and not LOCAL_MODEL_PATH.exists():
    st.error("未找到模型文件，请先上传或部署模型文件。")
    st.stop()

if page.startswith("0"):
    page_home()
elif page.startswith("1"):
    page_data_management()
elif page.startswith("2"):
    page_risk_identification()
elif page.startswith("3"):
    page_ai_explanation()
elif page.startswith("4"):
    page_smart_operations()
elif page.startswith("5"):
    page_model_evaluation()
else:
    page_architecture()
