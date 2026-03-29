from dataclasses import dataclass


@dataclass
class TrainingResult:
    n_samples: int
    n_train: int
    n_test: int
    n_bootstrap: int
