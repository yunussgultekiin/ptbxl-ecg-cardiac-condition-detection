import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchmetrics.functional.classification import multilabel_auroc, multilabel_confusion_matrix, multilabel_roc

BASE_FIGURE_SIZE = (12, 5)
SQUARE_FIGURE_SIZE = (8, 7)
CONFUSION_FIGURE_SIZE = (18, 4.5)
FIGURE_DPI = 110
PRIMARY_COLOR = "#2E5E8C"
SECONDARY_COLOR = "#E07A5F"
NEUTRAL_COLOR = "#9AA5B1"
SEQUENTIAL_CMAP = "Blues"
SUPERCLASS_COLORS = {"NORM": "#4C9F70", "MI": "#D1495B", "STTC": "#EDAE49", "CD": "#00798C", "HYP": "#7B5EA7"}
LOSS_COLORS = {"train": PRIMARY_COLOR, "val": SECONDARY_COLOR}
BINARY_LABELS = ("negative", "positive")

class PlotTheme:
    @staticmethod
    def apply():
        sns.set_theme(style="whitegrid", context="notebook")
        plt.rcParams.update({
            "figure.figsize": BASE_FIGURE_SIZE,
            "figure.dpi": FIGURE_DPI,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False
        })

    @staticmethod
    def class_colors(labels):
        return [SUPERCLASS_COLORS.get(label, NEUTRAL_COLOR) for label in labels]

    @staticmethod
    def figure(nrows=1, ncols=1, size=BASE_FIGURE_SIZE, sharex=False, sharey=False):
        return plt.subplots(nrows, ncols, figsize=size, sharex=sharex, sharey=sharey, constrained_layout=True)

    @staticmethod
    def decorate(ax, title, xlabel=None, ylabel=None):
        ax.set_title(title)
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")

class RunFigures:
    @staticmethod
    def roc_curve(labels, probabilities, class_names):
        PlotTheme.apply()
        figure, ax = PlotTheme.figure(size=SQUARE_FIGURE_SIZE)
        false_positive_rates, true_positive_rates, _ = multilabel_roc(probabilities, labels, num_labels=len(class_names))
        areas = multilabel_auroc(probabilities, labels, num_labels=len(class_names), average=None)
        curves = zip(class_names, PlotTheme.class_colors(class_names), false_positive_rates, true_positive_rates, areas)
        for class_name, color, false_positive_rate, true_positive_rate, area in curves:
            ax.plot(false_positive_rate.numpy(), true_positive_rate.numpy(), color=color, linewidth=1.6, label=f"{class_name} · AUC {area:.3f}")
        ax.plot([0, 1], [0, 1], color=NEUTRAL_COLOR, linestyle="--", linewidth=0.8)
        ax.legend(loc="lower right")
        PlotTheme.decorate(ax, f"ROC curves · macro AUROC {areas.mean():.3f}", "False positive rate", "True positive rate")
        return figure

    @staticmethod
    def confusion_matrix(labels, probabilities, thresholds, class_names):
        PlotTheme.apply()
        predictions = (probabilities >= thresholds).long()
        matrices = multilabel_confusion_matrix(predictions, labels, num_labels=len(class_names)).numpy()
        figure, axes = PlotTheme.figure(ncols=len(class_names), size=CONFUSION_FIGURE_SIZE)
        for ax, matrix, class_name, threshold in zip(axes, matrices, class_names, thresholds.tolist()):
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap=SEQUENTIAL_CMAP,
                cbar=False,
                square=True,
                linewidths=0.5,
                xticklabels=BINARY_LABELS,
                yticklabels=BINARY_LABELS,
                ax=ax
            )
            PlotTheme.decorate(ax, f"{class_name} · threshold {threshold:.2f}", "Predicted", "Actual")
        figure.suptitle("Confusion matrices", fontweight="bold")
        return figure

    @staticmethod
    def padded(curves):
        length = max(len(curve) for curve in curves)
        return np.array([np.pad(np.asarray(curve, dtype=float), (0, length - len(curve)), constant_values=np.nan) for curve in curves])

    @staticmethod
    def loss_curve(fold_histories, final_epoch):
        PlotTheme.apply()
        figure, ax = PlotTheme.figure()
        for split, color in LOSS_COLORS.items():
            curves = RunFigures.padded([history[split] for history in fold_histories])
            epochs = np.arange(1, curves.shape[1] + 1)
            mean, spread = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
            ax.plot(epochs, mean, color=color, linewidth=1.6, label=f"{split} loss")
            ax.fill_between(epochs, mean - spread, mean + spread, color=color, alpha=0.2)
        ax.axvline(final_epoch, color=NEUTRAL_COLOR, linestyle="--", linewidth=0.9, label=f"final epoch {final_epoch}")
        ax.legend()
        PlotTheme.decorate(ax, f"Train and validation loss across {len(fold_histories)} folds", "Epoch", "Loss")
        return figure