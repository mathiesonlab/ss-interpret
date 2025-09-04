import os
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, r2_score
import dtreeviz
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import load_metadata
from utils import SELECTED_STATS, apply_seed_to_path, colorlabels, get_model_preds_lf, get_stats, iterate_seeds

def get_X_y(pop, model, truth=False):
    X, _, valid_mask = get_stats(pop)

    if not truth:
        # average predictions from all seeds
        preds = []
        for seed in iterate_seeds(f"{pop}/{model}"):
            model_name = apply_seed_to_path(model, seed)
            p, _ = get_model_preds_lf(pop, model_name)
            preds.append(p)

        y = np.vstack(preds).mean(axis=0)
    else:
        # use truth labels
        y = load_metadata(pop)["label"].values

    # only non-nan ihs-maxabs (valid)
    y = y[valid_mask]

    assert X.shape[0] == y.shape[0], "X and y must be the same length"    
    return X, y

def sweep(pop, model, truth=False):
    results_df = pd.DataFrame(columns=["pop", "model", "depth", "mse", "r2", "acc", "f1"])

    X, y = get_X_y(pop, model, truth=truth)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train_bool = (y_train > 0.5).astype(int)
    y_test_bool = (y_test > 0.5).astype(int)

    # train model, sweep on depth
    # store metrics
    results = []
    for d in range(1, 20):
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
        results.append((pop, model, d, mse, r2, acc, f1))

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(results, columns=["pop", "model", "depth", "mse", "r2", "acc", "f1"]),
        ]
    )

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

    for i, metric in enumerate(metrics):
        axs[i].plot(
            results_df["depth"],
            results_df[metric],
        )
        axs[i].set_title(metric)
        axs[i].set_xlabel("Tree Depth")
        axs[i].set_ylabel(metric)
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(f"./figs/dtrees_sweep_{pop}_{model}_{'truth' if truth else 'modeloutput'}.png", 
                dpi=300, bbox_inches="tight")

def train_single_model(pop, d=3):

    model_names = [
        ("Truth", "truth"),
        ("New Output", "disc_N"),
    ]
    
    for model_name, model in model_names:
        print("-" * 10 + f" {model_name}/averaged/d={d} " + "-" * 10)

        X, y = get_X_y(pop, model, truth=(model_name == "Truth"))

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        y_train_bool = (y_train > 0.5).astype(int)
        y_test_bool = (y_test > 0.5).astype(int)

        # train model, sweep on depth
        # store metrics
        print(f"Fitting dtree model with max depth={d}")
        # regressor
        model_reg = DecisionTreeRegressor(max_depth=d, random_state=42)
        model_reg.fit(X_train, y_train)

        # classifier
        model_clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        model_clf.fit(X_train, y_train_bool)

        y_pred = model_reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        y_pred_bool = model_clf.predict(X_test)
        acc = np.mean(y_pred_bool == y_test_bool)
        f1 = f1_score(y_test_bool, y_pred_bool)
        print(f"Metrics: mse={mse}, r2={r2}, acc={acc}, f1={f1}")

        # dtreeviz visualizations
        viz = dtreeviz.model(
            model_reg,
            X_train[:1000],
            y_train[:1000],
            target_name="Predicted" if model_name != "Truth" else "Truth",
            feature_names=SELECTED_STATS,
        )
        v = viz.view(fontname="monospace", title="") #f"{pop} - {model_name}")

        viz2 = dtreeviz.model(
            model_clf,
            X_train,
            y_train_bool,
            target_name="Predicted",
            feature_names=SELECTED_STATS,
            class_names=["Generated", "Real"],
        )
        v2 = viz2.view(fontname="monospace", title="") #f"{pop} - {model_name}")

        # Sklearn tree visualizations
        if model_name != "Truth":
            # only a regressor tree

            from sklearn import tree
            fig_reg, ax_reg = plt.subplots(figsize=(12, 8))
            tree.plot_tree(model_reg, feature_names=SELECTED_STATS, ax=ax_reg, filled=True)
            plt.title("") #f"{pop} {model_name}")
            fig_reg.savefig(f"./figs/dtree_reg_{pop}_{model_name}.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig_reg)

        # always a classifier tree
        v2.save(f"./figs/dtree_clf_{pop}_{model_name}.svg")
        os.remove(f"./figs/dtree_clf_{pop}_{model_name}")


def feature_importance(pop, d=10):
    """
    Train single-seed random forests for a range of seeds and combine feature importances.
    Summarize with a plot of mean feature importances and error bars (std) across seeds.
    """
    model_labels = [
        ("Truth", "truth"),
        ("New Output", f"disc_N"),
        # ("GAN Output", f"{pop}_N_230410"),
    ]
    runs = 3

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    fig.subplots_adjust(hspace=0.4)
    for idx, (label, model) in enumerate(model_labels):
        all_importances = []
        X, y = get_X_y(pop, model, truth=(label == "Truth"))
        for seed in range(runs):
            print(f"Fitting random forest {seed} for {label}")
            X_train = X
            y_train = y

            if label == "Truth":
                rf = RandomForestClassifier(max_depth=d, max_features="log2", 
                                            random_state=seed, n_estimators=100, n_jobs=24)
            else:
                rf = RandomForestRegressor(max_depth=d, max_features="log2",
                                        random_state=seed, n_estimators=100, n_jobs=24)

            rf.fit(X_train, y_train)
            importances = rf.feature_importances_
            
            print(f"(estimate) num parameters: {sum(tree.tree_.node_count for tree in rf.estimators_) * 5}")
            
            if label == "Truth":
                acc = rf.score(X_train, y_train)
                print(f"Accuracy: {acc:.4f}")
                metricstr = f"Acc: {acc:.3f}"
            else:
                mse = mean_squared_error(y_train, rf.predict(X_train))
                print(f"MSE: {mse:.4f}")
                metricstr = f"MSE: {mse:.3f}"

            for fname, imp in zip(SELECTED_STATS, importances):
                all_importances.append({
                    "Feature": fname,
                    "Importance": imp,
                    "Seed": seed
                })

        df_importances = pd.DataFrame(all_importances)
        # Only plot top N features by mean importance
        top_features = (
            df_importances.groupby("Feature")["Importance"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .index
        )
        df_plot = df_importances[df_importances["Feature"].isin(top_features)]

        ax = axs[idx]
        sns.barplot(
            data=df_plot,
            x="Feature",
            y="Importance",
            errorbar="sd",
            capsize=0.1,
            err_kws={'linewidth': 1.5},
            order=top_features,
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        colorlabels(ax.get_xticklabels())
        ax.set_title(f"{label} | {metricstr}")
        ax.set_ylabel("Mean Decrease in Impurity" if label == "Truth" else 
                      "Mean Decrease in Variance")
        ax.set_xlabel("")

    fig.suptitle(
        pop,
        y=0.92,
        x=0.1,
        fontsize=16,
        bbox=dict(facecolor='white', edgecolor='black', linewidth=1)
    )
    plt.savefig(f"./figs/feature_importance_{pop}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dtree.py <pop> <forest | single>")
        sys.exit(1)
    
    pop = sys.argv[1]
    assert sys.argv[2] in ["forest", "single"]

    if sys.argv[2] == "forest":
        # plot feature importance for rf
        feature_importance(pop, 10)
    else:
        # plot single model dtree
        train_single_model(pop, 3)