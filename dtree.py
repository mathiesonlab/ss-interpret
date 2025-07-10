import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, r2_score
import dtreeviz
import matplotlib.pyplot as plt

STATS = (
    [f"SFS_{i}" for i in range(1, 11)]
    + [f"inter-SNP_{i}" for i in range(1, 37)]
    + [f"LD_{i}" for i in range(1, 16)]
    + ["$\pi$", "#haps"]
)
EXTRA_STATS = [
    "ihs_maxabs",
    "tajimas_d",
    "garud_h1",
    "garud_h12",
    "garud_h123",
    "garud_h2_h1",
]
ALL_STATS = STATS + EXTRA_STATS

TRAIN_POP = "CEU"
TEST_POP = "GBR"
ORIG_MODEL = "230410"
REDU_MODEL = "250626"
SEEDS = list(range(0, 5))

STATS_PATH = "./data/summary_stats/stats_{TEST_POP}.npy"
PREDS_PATH = "./results/predictions/{TRAIN_POP}_{SEED}_{MODEL}_{TEST_POP}.txt"

MODELS = [ORIG_MODEL, REDU_MODEL]

VIZ = False

results_df = pd.DataFrame(columns=["model", "type", "depth", "mse", "r2", "acc", "f1"])
for model, model_name in zip(MODELS, ["original", "reduced"]):
    for average, avg_name in zip([True, False], ["averaged", "full"]):
        print("-" * 10 + f" {model_name}/{avg_name} " + "-" * 10)
        spath = STATS_PATH.format(TEST_POP=TEST_POP)

        # concat all data
        X = np.zeros((0, 63))
        y = np.array([])
        n = np.load(spath).shape[0]
        for seed in SEEDS:
            ppath = PREDS_PATH.format(
                TRAIN_POP=TRAIN_POP, SEED=seed, MODEL=model, TEST_POP=TEST_POP
            )

            X = np.concatenate([X, np.load(spath)])
            y = np.concatenate([y, np.loadtxt(ppath)[:, -1]])

        if average:
            X = X[:n]
            y = np.mean(np.reshape(y, (n, -1), order="F"), axis=1)

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        y_train_bool = (y_train > 0.5).astype(int)
        y_test_bool = (y_test > 0.5).astype(int)

        # train model, sweep on depth
        # store metrics
        results = []
        for d in range(3, 20):
            print(f"Fitting dtree model with max depth={d}")
            # regressor
            model_reg = DecisionTreeRegressor(max_depth=d, random_state=42)
            model_reg.fit(X_train, y_train)

            # classifier
            model_clf = DecisionTreeClassifier(max_depth=d, random_state=42)
            model_clf.fit(X_train, y_train_bool)

            # evaluate model
            y_pred = model_reg.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            y_pred_bool = model_clf.predict(X_test)
            acc = np.mean(y_pred_bool == y_test_bool)
            f1 = f1_score(y_test_bool, y_pred_bool)
            results.append((model_name, avg_name, d, mse, r2, acc, f1))

        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(results, columns=["model", "type", "depth", "mse", "r2", "acc", "f1"]),
            ]
        )

        if VIZ:
            viz = dtreeviz.model(
                model_reg,
                X_train[:5000],
                y_train[:5000],
                target_name="Predicted",
                feature_names=STATS,
            )

            viz2 = dtreeviz.model(
                model_clf,
                X_train,
                y_train_bool,
                target_name="Predicted",
                feature_names=STATS,
                class_names=["Generated", "Real"],
            )

            viz.view()

with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.max_colwidth",
    None,
):
    print(results_df)

metrics = ["mse", "r2", "acc", "f1"]
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

colors = {"original": "blue", "reduced": "orange"}
linestyles = {"averaged": "solid", "full": "dashed"}

for i, metric in enumerate(metrics):
    for m in results_df["model"].unique():
        for t in results_df["type"].unique():
            df = results_df[(results_df["model"] == m) & (results_df["type"] == t)]
            axs[i].plot(
                df["depth"],
                df[metric],
                label=f"{m} ({t})",
                color=colors[m],
                linestyle=linestyles[t],
            )
    axs[i].set_title(metric)
    axs[i].set_xlabel("Tree Depth")
    axs[i].set_ylabel(metric)
    axs[i].legend()

plt.tight_layout()
plt.savefig("./results/figs/dtrees.png", dpi=300, bbox_inches="tight")
