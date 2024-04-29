import comet_ml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction

def compute_metrics(pred: EvalPrediction):
    
    experiment = comet_ml.get_global_experiment()

    labels = pred.label_ids

    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(y_true=labels, y_pred=preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_pred=labels, y_true=preds, average="weighted"
    )
    
    if experiment:
        epoch = int(experiment.curr_epoch) if experiment.curr_epoch is not None else 0
        experiment.set_epoch(epoch)
        experiment.log_confusion_matrix(
            y_true=labels,
            y_predicted=preds,
            file_name=f"confusion-matrix-epoch-{epoch}.json",
            labels=["non-disaster", "disaster"],
        )
    

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1-score": f1}
