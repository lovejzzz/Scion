from pathlib import Path

from scion.training import parse_training_metrics


def test_parse_training_metrics(tmp_path: Path) -> None:
    log = tmp_path / "training.log"
    log.write_text(
        "Iter 1: Val loss 15.306, Val took 1.0s\n"
        "Iter 400: Val loss 11.124, Val took 1.0s\n"
        "Iter 400: Train loss 11.297, Learning Rate 1e-5, Trained Tokens 119818, "
        "Peak mem 92.463 GB\n"
        "Test loss 11.262, Test ppl 77815.093.\n",
        encoding="utf-8",
    )

    assert parse_training_metrics(log) == {
        "validationLoss": [
            {"iteration": 1, "loss": 15.306},
            {"iteration": 400, "loss": 11.124},
        ],
        "finalTrain": {
            "iteration": 400,
            "loss": 11.297,
            "trainedTokens": 119818,
            "reportedPeakMemoryGb": 92.463,
        },
        "test": {"loss": 11.262, "perplexity": 77815.093},
    }
