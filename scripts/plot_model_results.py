import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def plot_metric_by_model(data, feature_set, metric_key, out_dir):
    ranking = data['by_feature_set'][feature_set]['ranking']
    models = [r['model'] for r in ranking]
    values = [r[metric_key] for r in ranking]

    plt.figure(figsize=(8, 4.5))
    if sns is not None:
        sns.barplot(x=values, y=models, palette='viridis')
    else:
        plt.barh(models, values, color='steelblue')
    plt.xlabel(metric_key.replace('_', ' ').title())
    plt.title(f"{metric_key.replace('_', ' ').title()} - {feature_set}")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{feature_set}_{metric_key}.png")
    plt.savefig(out_path)
    plt.close()
    print('Saved', out_path)


def plot_overall_comparison(data, metric_key, out_dir):
    feature_sets = sorted(data['by_feature_set'].keys())
    models = []
    for content in data['by_feature_set'].values():
        for r in content['ranking']:
            if r['model'] not in models:
                models.append(r['model'])

    values_by_fs = {}
    for fs in feature_sets:
        ranking = data['by_feature_set'][fs]['ranking']
        values_by_fs[fs] = [next(r[metric_key] for r in ranking if r['model'] == model) for model in models]

    plt.figure(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.7 / max(1, len(feature_sets))
    for i, fs in enumerate(feature_sets):
        plt.bar(x + i * width, values_by_fs[fs], width=width, label=fs)

    plt.xticks(x + width * (len(feature_sets) - 1) / 2, models, rotation=45, ha='right')
    plt.ylabel(metric_key.replace('_', ' ').title())
    plt.title(f"Model comparison by {metric_key}")
    plt.legend(title='Feature set')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"all_models_{metric_key}.png")
    plt.savefig(out_path)
    plt.close()
    print('Saved', out_path)


def plot_confusion_matrix(metrics, out_dir):
    labels = metrics.get('labels')
    cm = np.array(metrics.get('confusion_matrix'))
    plt.figure(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    else:
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, int(cm[i, j]), ha='center', va='center', color='white')
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Best Model')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'confusion_matrix_best_model.png')
    plt.savefig(out_path)
    plt.close()
    print('Saved', out_path)


def plot_train_holdout_cv_metrics(metrics, out_dir):
    labels = ['Train', 'Holdout', 'CV mean']
    values = [metrics.get('train_f1_macro'), metrics.get('f1_macro'), metrics.get('cv_f1_macro_mean')]

    plt.figure(figsize=(7, 4.5))
    if sns is not None:
        sns.barplot(x=values, y=labels, palette='coolwarm')
    else:
        plt.barh(labels, values, color=['#4c72b0', '#55a868', '#c44e52'])
    for i, v in enumerate(values):
        plt.text(v + 0.005, i, f'{v:.3f}', va='center')
    plt.xlim(0, 1)
    plt.xlabel('F1 Macro')
    plt.title('Train / Holdout / CV F1 Macro')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'train_holdout_cv_f1_macro.png')
    plt.savefig(out_path)
    plt.close()
    print('Saved', out_path)


def main():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / 'models'
    ensure_dir(models_dir / 'plots')

    comparison_path = models_dir / 'model_comparison.json'
    metrics_path = models_dir / 'metrics.json'

    comparison = load_json(comparison_path)
    metrics = load_json(metrics_path)

    out_dir = str(models_dir / 'plots')

    # Per-feature-set charts
    for fs in comparison['by_feature_set'].keys():
        plot_metric_by_model(comparison, fs, 'test_accuracy', out_dir)
        plot_metric_by_model(comparison, fs, 'test_f1_macro', out_dir)

    # Overall comparison charts
    plot_overall_comparison(comparison, 'test_accuracy', out_dir)
    plot_overall_comparison(comparison, 'test_f1_macro', out_dir)

    # Train / Holdout / CV summary for the final model
    plot_train_holdout_cv_metrics(metrics, out_dir)

    # Confusion matrix for the reported best model metrics
    plot_confusion_matrix(metrics, out_dir)


if __name__ == '__main__':
    main()
