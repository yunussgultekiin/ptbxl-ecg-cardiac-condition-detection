import torch
from torchmetrics.functional.classification import multilabel_auroc, multilabel_average_precision, multilabel_f1_score, multilabel_precision, multilabel_recall

THRESHOLD_GRID = torch.round(torch.arange(0.05, 0.951, 0.01), decimals=2)
MONITOR_THRESHOLD = 0.5
BOOTSTRAP_METRICS = ("auroc", "auprc", "precision", "recall", "f1")
BOOTSTRAP_CHUNK_SIZE = 100
BOOTSTRAP_METHOD = "patient_cluster_bootstrap"
INTERVAL_METHOD = "percentile"
THRESHOLD_SOURCE = "oof"
UNDEFINED_REPLICATES_RULE = "skipped per class when the replicate has no positive or no negative records; macro requires every class"

class ThresholdSearch:
    @staticmethod
    def fit(labels, probabilities):
        class_count = labels.shape[1]
        f1_grid = torch.stack([
            multilabel_f1_score(probabilities, labels, num_labels=class_count, threshold=float(threshold), average=None)
            for threshold in THRESHOLD_GRID
        ])
        return THRESHOLD_GRID[f1_grid.argmax(dim=0)]

class MultilabelScores:
    @staticmethod
    def ranking(labels, probabilities, class_names):
        auroc = multilabel_auroc(probabilities, labels, num_labels=len(class_names), average=None)
        auprc = multilabel_average_precision(probabilities, labels, num_labels=len(class_names), average=None)
        per_class = {f"auroc_{name}": value.item() for name, value in zip(class_names, auroc)}
        return {"macro_auroc": auroc.mean().item(), "macro_auprc": auprc.mean().item()} | per_class

    @staticmethod
    def thresholded(labels, probabilities, thresholds, class_names):
        predictions = (probabilities >= thresholds).long()
        scores = {}
        for metric, function in (("precision", multilabel_precision), ("recall", multilabel_recall), ("f1", multilabel_f1_score)):
            values = function(predictions, labels, num_labels=len(class_names), average=None)
            scores |= {f"macro_{metric}": values.mean().item()} | {f"{metric}_{name}": value.item() for name, value in zip(class_names, values)}
        return scores

    @staticmethod
    def summary(labels, probabilities, class_names, thresholds=None):
        scores = MultilabelScores.ranking(labels, probabilities, class_names)
        return scores if thresholds is None else scores | MultilabelScores.thresholded(labels, probabilities, thresholds, class_names)

    @staticmethod
    def class_average_precision(labels, probabilities, class_names):
        values = multilabel_average_precision(probabilities, labels, num_labels=len(class_names), average=None)
        return {f"auprc_{name}": value.item() for name, value in zip(class_names, values)}

    @staticmethod
    def monitoring(labels, probabilities, class_names):
        return MultilabelScores.summary(labels, probabilities, class_names, MONITOR_THRESHOLD)

class WeightedScores:
    @staticmethod
    def divide(numerator, denominator):
        defined = denominator > 0
        return torch.where(defined, numerator / torch.where(defined, denominator, torch.ones_like(denominator)), torch.zeros_like(numerator))

    @staticmethod
    def thresholded(weights, labels, predictions):
        positives = labels.double()
        predicted = predictions.double()
        true_positive = weights @ (positives * predicted)
        false_positive = weights @ ((1 - positives) * predicted)
        false_negative = weights @ (positives * (1 - predicted))
        return {
            "precision": WeightedScores.divide(true_positive, true_positive + false_positive),
            "recall": WeightedScores.divide(true_positive, true_positive + false_negative),
            "f1": WeightedScores.divide(2 * true_positive, 2 * true_positive + false_positive + false_negative)
        }

    @staticmethod
    def ranking(weights, labels, probabilities):
        order = torch.argsort(probabilities, descending=True, stable=True)
        scores = probabilities[order]
        positives = labels[order].double()
        group_ends = torch.cat([scores[1:] != scores[:-1], torch.ones(1, dtype=torch.bool)])
        ordered_weights = weights[:, order]
        true_positive = torch.cumsum(ordered_weights * positives, dim=1)[:, group_ends]
        false_positive = torch.cumsum(ordered_weights * (1 - positives), dim=1)[:, group_ends]
        start = torch.zeros(weights.shape[0], 1, dtype=torch.float64)
        true_positive_rate = WeightedScores.divide(torch.cat([start, true_positive], dim=1), true_positive[:, -1:])
        false_positive_rate = WeightedScores.divide(torch.cat([start, false_positive], dim=1), false_positive[:, -1:])
        auroc = ((false_positive_rate[:, 1:] - false_positive_rate[:, :-1]) * (true_positive_rate[:, 1:] + true_positive_rate[:, :-1]) / 2).sum(dim=1)
        precision = WeightedScores.divide(true_positive, true_positive + false_positive)
        auprc = ((true_positive_rate[:, 1:] - true_positive_rate[:, :-1]) * precision).sum(dim=1)
        return auroc, auprc, true_positive[:, -1] > 0, false_positive[:, -1] > 0

    @staticmethod
    def replicates(weights, labels, probabilities, thresholds):
        scores = WeightedScores.thresholded(weights, labels, probabilities >= thresholds)
        rankings = [WeightedScores.ranking(weights, labels[:, index], probabilities[:, index]) for index in range(labels.shape[1])]
        scores["auroc"] = torch.stack([ranking[0] for ranking in rankings], dim=1)
        scores["auprc"] = torch.stack([ranking[1] for ranking in rankings], dim=1)
        valid = torch.stack([ranking[2] & ranking[3] for ranking in rankings], dim=1)
        return scores, valid

class PatientBootstrap:
    def __init__(self, replicates, confidence_level, seed):
        self.replicates = replicates
        self.confidence_level = confidence_level
        self.seed = seed

    @property
    def quantiles(self):
        tail = (1 - self.confidence_level) / 2
        return torch.tensor([tail, 1 - tail], dtype=torch.float64)

    def weight_chunks(self, patient_ids):
        _, patient_index = torch.unique(torch.as_tensor(patient_ids).cpu(), return_inverse=True)
        patient_count = int(patient_index.max()) + 1
        generator = torch.Generator().manual_seed(self.seed)
        for start in range(0, self.replicates, BOOTSTRAP_CHUNK_SIZE):
            size = min(BOOTSTRAP_CHUNK_SIZE, self.replicates - start)
            draws = torch.randint(patient_count, (size, patient_count), generator=generator)
            counts = torch.zeros(size, patient_count, dtype=torch.float64).scatter_add_(1, draws, torch.ones(size, patient_count, dtype=torch.float64))
            yield counts[:, patient_index]

    def replicate_scores(self, labels, probabilities, patient_ids, thresholds):
        chunks = [WeightedScores.replicates(weights, labels, probabilities, thresholds) for weights in self.weight_chunks(patient_ids)]
        scores = {metric: torch.cat([chunk[0][metric] for chunk in chunks]) for metric in BOOTSTRAP_METRICS}
        return scores, torch.cat([chunk[1] for chunk in chunks])

    def estimate(self, value, samples):
        if len(samples) == 0:
            return {"value": value, "lower": None, "upper": None}
        lower, upper = torch.quantile(samples, self.quantiles).tolist()
        return {"value": value, "lower": lower, "upper": upper}

    def section(self, labels, probabilities, patient_ids, thresholds, class_names, scores):
        labels, probabilities, thresholds = labels.cpu().long(), probabilities.cpu().float(), thresholds.cpu().float()
        patient_ids = torch.as_tensor(patient_ids).cpu()
        point = MultilabelScores.summary(labels, probabilities, class_names, thresholds) | MultilabelScores.class_average_precision(labels, probabilities, class_names) | scores
        replicate_scores, valid = self.replicate_scores(labels, probabilities, patient_ids, thresholds)
        macro_valid = valid.all(dim=1)
        return {
            "records": len(labels),
            "patients": len(set(patient_ids.tolist())),
            "valid_replicates": {"macro": int(macro_valid.sum())} | {name: int(valid[:, index].sum()) for index, name in enumerate(class_names)},
            "macro": {metric: self.estimate(point[f"macro_{metric}"], replicate_scores[metric][macro_valid].mean(dim=1)) for metric in BOOTSTRAP_METRICS},
            "classes": {
                name: {metric: self.estimate(point[f"{metric}_{name}"], replicate_scores[metric][valid[:, index], index]) for metric in BOOTSTRAP_METRICS}
                for index, name in enumerate(class_names)
            }
        }

    def report(self, sections, thresholds, class_names, cluster_column):
        return {
            "method": BOOTSTRAP_METHOD,
            "interval": INTERVAL_METHOD,
            "replicates": self.replicates,
            "confidence_level": self.confidence_level,
            "seed": self.seed,
            "cluster_column": cluster_column,
            "undefined_replicates": UNDEFINED_REPLICATES_RULE,
            "thresholds": {"source": THRESHOLD_SOURCE, "values": {name: round(threshold, 2) for name, threshold in zip(class_names, thresholds.tolist())}},
            "sections": {name: self.section(*arrays, thresholds, class_names, scores) for name, (*arrays, scores) in sections.items()}
        }