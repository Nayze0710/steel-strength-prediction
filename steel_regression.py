# =============================================================================
# 文件名称：steel_regression.py
#
# 一、研究目标
#   利用 SteelBench v1.0 公开数据集，以钢材的化学成分与热处理工艺为输入，
#   建立回归模型预测钢材的抗拉强度（tensile_strength，单位 MPa）。
#   本程序真实读取原始 CSV，不使用任何编造数据，也不修改原始数据文件。
#
# 二、变量与单位
#   输入特征（11 个）：
#     C、Mn、Si、Cr、Ni、Mo、V、Cu、Al —— 化学成分质量分数，单位 wt.%
#     austenitize_T —— 奥氏体化温度，单位 °C
#     temper_T      —— 回火温度，单位 °C
#   预测目标：
#     tensile_strength —— 抗拉强度，单位 MPa
#
# 三、输出文件（均写入 results/ 目录）
#     results/metrics.csv                评价指标（MAE、RMSE、R²）
#     results/predictions.csv            测试集样本编号、来源、参考强度与预测强度
#     results/prediction_comparison.png  参考值与预测值对比图（含 y = x 参考线）
#     results/feature_importance.png     随机森林特征重要性图
#
# 四、运行方法
#   在项目根目录执行：python steel_regression.py
#   程序内部全部使用相对路径（data/ 与 results/），不依赖绝对路径。
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 数据文件路径（相对项目根目录），原始 CSV 只读，不做任何修改
DATA_PATH = os.path.join("data", "steelbench_core_open.csv")
# 结果输出目录，程序启动时若不存在会自动创建
OUT_DIR = "results"
# 预测目标列名：抗拉强度（MPa）
TARGET = "tensile_strength"
# 输入特征列名：9 个化学成分（wt.%）+ 2 个热处理温度（°C）
# 注意：屈服强度、延伸率、硬度等其他力学性能，以及 grade_id、source
# 均不进入特征集合，避免目标泄漏与信息泄漏
FEATURES = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "Cu", "Al",
            "austenitize_T", "temper_T"]
# 全局随机种子：保证数据划分、随机森林初始化可复现
RANDOM_STATE = 42


def load_data(path):
    # 读取原始 CSV 文件；只读不写，保证原始数据不被改动
    df = pd.read_csv(path)
    # 打印原始数据形状（行数、列数）与全部列名，便于核对字段是否与说明一致
    print("[1] raw data shape:", df.shape)
    print("[1] raw columns:", list(df.columns))
    return df


def inspect_data(df):
    # 数据探查：在建模前检查样本量、钢种数量、单位、缺失值与重复值
    print("\n[2] data inspection")
    # 样本数量（行数）
    print("sample count:", len(df))
    # 不同钢种（grade_id）的数量，后续按钢种分组划分数据集
    print("unique grade_id:", df["grade_id"].nunique())
    # 明确目标变量的物理单位
    print("target unit: tensile_strength in MPa")
    # 按 data_tier 统计记录数，区分数据来源性质
    print("\ndata_tier counts:")
    tier_counts = df["data_tier"].value_counts()
    for tier, n in tier_counts.items():
        print("  %s: %d" % (tier, n))
    # 实测数据：NIMS 与 Kaggle 两条来源为实验室实测值
    measured = df[df["data_tier"].isin(["nims_measured", "kaggle_measured"])]
    # 规范最低值：EMK 手册给出的是技术规范下限，不等于实测强度
    spec = df[df["data_tier"] == "emk_spec_verified"]
    print("measured strength rows:", len(measured))
    print("specification-minimum rows:", len(spec))
    # 逐列缺失值统计（原始数据，不填补）
    print("\nmissing values per column (raw):")
    print(df.isna().sum().to_string())
    # 完全重复记录数
    print("\nexact duplicate rows:", int(df.duplicated().sum()))
    # 目标缺失与钢种编号缺失的记录数
    print("rows with missing target:", int(df[TARGET].isna().sum()))
    print("rows with missing grade_id:", int(df["grade_id"].isna().sum()))


def clean_data(df):
    # 数据清洗：只剔除无效记录，不做任何数据补造
    print("\n[3] data cleaning")
    n0 = len(df)
    # 剔除目标缺失或钢种编号缺失的记录（这两类记录无法用于建模）
    df = df.dropna(subset=[TARGET, "grade_id"]).copy()
    n1 = len(df)
    # 剔除抗拉强度非正值的异常记录（强度必须为正值才符合物理意义）
    df = df[df[TARGET] > 0].copy()
    n2 = len(df)
    # 剔除完全重复的记录，保持一条唯一记录
    df = df.drop_duplicates().reset_index(drop=True)
    n3 = len(df)
    # 依次打印各步剩余记录数，报告清洗前后的数量变化
    print("start rows:", n0)
    print("after dropping missing/invalid target and grade_id:", n1)
    print("after dropping non-positive target:", n2)
    print("after dropping exact duplicates:", n3)
    print("removed total:", n0 - n3)
    return df


def split_data(df):
    # 数据集划分：按钢种（grade_id）分组划分，防止同一钢种同时出现在训练集与测试集
    print("\n[4] grouped split by grade_id (GroupShuffleSplit)")
    # n_splits=1 表示只做一次划分；test_size=0.2 表示约 20% 的钢种进入测试集
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["grade_id"]))
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    # 取出训练集与测试集包含的钢种集合
    train_grades = set(train["grade_id"])
    test_grades = set(test["grade_id"])
    # 求交集，检查是否存在钢种重叠（若有重叠说明发生了数据泄漏）
    overlap = train_grades & test_grades
    print("train rows / test rows:", len(train), "/", len(test))
    print("train grades / test grades:", len(train_grades), "/", len(test_grades))
    print("overlapping grades:", len(overlap))
    # 断言：两组钢种必须完全不重叠，否则直接报错终止
    assert len(overlap) == 0, "grade leakage between train and test"
    # 打印测试集中各数据来源的样本数，便于后续分来源评估误差
    print("test data_tier counts:")
    print(test["data_tier"].value_counts().to_string())
    return train, test


def resolve_features(train):
    # 检查候选特征在训练集中的可用性：整列完全缺失的特征必须剔除
    print("\n[5] feature availability check on training set")
    dropped = [c for c in FEATURES if train[c].isna().all()]
    if dropped:
        print("columns entirely missing in train, excluded:", dropped)
    else:
        print("columns entirely missing in train: none")
    # 实际参与建模的特征列表
    used = [c for c in FEATURES if c not in dropped]
    print("features used:", used)
    return used


def build_models(features):
    # 构建两个回归模型，并统一封装为 Pipeline：
    # 第一步：缺失值填补（中位数），只在训练集上拟合，避免用到测试集信息
    # add_indicator=True 会额外生成“该字段是否缺失”的指示变量，
    # 用于保留缺失这一信息（未知成分/温度绝不填 0 这种物理零值）
    linear = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", LinearRegression()),
    ])
    # 第二步：随机森林回归，100 棵树，随机种子 42，保证结果可复现
    forest = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", RandomForestRegressor(n_estimators=100,
                                        random_state=RANDOM_STATE,
                                        n_jobs=-1)),
    ])
    # 以字典形式返回，键名即模型名称，会写入结果文件
    return {"LinearRegression": linear, "RandomForest": forest}


def evaluate(name, y_true, y_pred):
    # 计算三个回归评价指标：
    # MAE  —— 平均绝对误差，单位与目标一致（MPa），越小越好
    # RMSE —— 均方根误差，单位 MPa，对较大误差更敏感
    # R²   —— 决定系数，表示模型对目标变异的解释比例，越接近 1 越好
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    # 创建结果目录（已存在则跳过）
    os.makedirs(OUT_DIR, exist_ok=True)

    # 第 1 步：读取原始数据
    df = load_data(DATA_PATH)
    # 第 2 步：数据探查（列名、样本量、单位、缺失、重复、来源分布）
    inspect_data(df)
    # 第 3 步：数据清洗
    df = clean_data(df)
    # 第 4 步：按钢种分组划分训练集 / 测试集
    train, test = split_data(df)
    # 第 5 步：确定最终参与建模的特征
    features = resolve_features(train)

    # 第 6 步：只用训练集训练模型；测试集不参与任何调参
    print("\n[6] training models (test set never used for tuning)")
    models = build_models(features)
    fitted = {}
    for name, pipe in models.items():
        pipe.fit(train[features], train[TARGET])
        fitted[name] = pipe
        print("trained:", name)

    # 第 7 步：在测试集上预测并计算指标
    print("\n[7] test-set metrics")
    y_true = test[TARGET].to_numpy()
    # 分别得到两个模型在测试集上的预测值
    preds = {name: pipe.predict(test[features]) for name, pipe in fitted.items()}

    # 汇总指标表：每个模型既给出整体指标，也给出分数据来源的指标
    rows = []
    for name, y_pred in preds.items():
        mae, rmse, r2 = evaluate(name, y_true, y_pred)
        # 整体（全部测试样本）指标
        rows.append({"model": name, "subset": "test_all",
                     "n_samples": len(test), "MAE_MPa": round(mae, 2),
                     "RMSE_MPa": round(rmse, 2), "R2": round(r2, 4)})
        print("%-16s MAE=%.2f  RMSE=%.2f  R2=%.4f" % (name, mae, rmse, r2))

        # 分数据来源（data_tier）统计误差：实测数据与规范最低值不能混为一谈
        for tier in sorted(test["data_tier"].unique()):
            mask = (test["data_tier"] == tier).to_numpy()
            # 样本太少时不做统计，避免指标失真
            if mask.sum() < 2:
                continue
            m_mae, m_rmse, m_r2 = evaluate(name, y_true[mask], y_pred[mask])
            rows.append({"model": name, "subset": "test_tier:" + tier,
                         "n_samples": int(mask.sum()),
                         "MAE_MPa": round(m_mae, 2),
                         "RMSE_MPa": round(m_rmse, 2),
                         "R2": round(m_r2, 4)})
            print("  tier %-18s n=%4d MAE=%.2f RMSE=%.2f R2=%.4f"
                  % (tier, mask.sum(), m_mae, m_rmse, m_r2))

    # 第 8 步：把评价指标写入 results/metrics.csv
    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(OUT_DIR, "metrics.csv")
    # utf-8-sig 便于 Excel 正确显示中文
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print("\n[8] saved:", metrics_path)

    # 把测试集逐样本的预测结果写入 results/predictions.csv
    # 字段含义：样本编号 heat_id、数据来源 source、来源性质 data_tier、
    #          参考强度（实测或规范值）、两个模型的预测强度
    pred_df = pd.DataFrame({
        "heat_id": test["heat_id"].values,
        "source": test["source"].values,
        "data_tier": test["data_tier"].values,
        "tensile_strength_reference_MPa": test[TARGET].values,
    })
    for name, y_pred in preds.items():
        pred_df["predicted_" + name + "_MPa"] = np.round(y_pred, 2)
    pred_path = os.path.join(OUT_DIR, "predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print("[8] saved:", pred_path, "rows:", len(pred_df))

    # 绘制对比图：左右两个子图分别是线性回归与随机森林的
    # “参考强度 vs 预测强度”散点图，红色虚线 y = x 表示理想预测
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (name, y_pred) in zip(axes, preds.items()):
        ax.scatter(y_true, y_pred, s=18, alpha=0.65, edgecolors="none")
        # 计算图中 y = x 参考线的取值范围
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        # y = x 参考线：点落在线上表示预测值与参考值完全一致
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")
        ax.set_xlabel("Reference tensile strength (MPa)")
        ax.set_ylabel("Predicted tensile strength (MPa)")
        # 在子图标题中标注该模型的三个指标，便于直接比较
        mae, rmse, r2 = evaluate(name, y_true, y_pred)
        ax.set_title("%s  MAE=%.1f  RMSE=%.1f  R2=%.3f" % (name, mae, rmse, r2))
        ax.legend(loc="upper left")
    fig.suptitle("Prediction vs reference on group-split test set")
    fig.tight_layout()
    cmp_path = os.path.join(OUT_DIR, "prediction_comparison.png")
    fig.savefig(cmp_path, dpi=150)
    plt.close(fig)
    print("[8] saved:", cmp_path)

    # 绘制随机森林特征重要性：数值表示该特征在树分裂中降低不纯度的贡献
    # 注意：Pipeline 中的填补器会为缺失字段追加“缺失指示”变量，
    # 因此需要把指示变量的重要性归并回对应的原始特征后再排序展示
    rf = fitted["RandomForest"].named_steps["model"]
    imputer = fitted["RandomForest"].named_steps["imputer"]
    # 取出填补器输出的特征名（原始特征 + missingindicator_ 前缀的指示变量）
    out_names = list(imputer.get_feature_names_out(features))
    # 去掉缺失指示前缀，得到每个输出列对应的原始特征名
    base_names = [n.replace("missingindicator_", "") for n in out_names]
    imp_all = pd.Series(rf.feature_importances_, index=base_names)
    # 按原始特征分组求和：原始特征的重要性 = 其本身 + 对应缺失指示变量
    imp = imp_all.groupby(level=0).sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    imp.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Random forest feature importance")
    ax.set_title("Feature importance (RandomForest, 100 trees)")
    fig.tight_layout()
    imp_path = os.path.join(OUT_DIR, "feature_importance.png")
    fig.savefig(imp_path, dpi=150)
    plt.close(fig)
    print("[8] saved:", imp_path)

    # 在控制台按重要性从大到小打印各特征，便于在 README 中讨论
    print("\n[9] feature importance:")
    for k, v in imp.sort_values(ascending=False).items():
        print("  %-15s %.4f" % (k, v))


# 脚本入口：直接运行本文件时执行 main()，被导入时不自动运行
if __name__ == "__main__":
    main()

