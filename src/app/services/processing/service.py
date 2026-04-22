from sklearn.pipeline import Pipeline

from .age_binner.transformer import AgeBinner
from .cholesterol_missingness.transformer import CholesterolMissingness
from .column_selector.transformer import ColumnSelector
from .imputer.transformer import Imputer
from .scaler.transformer import Scaler
from .waist_to_height_ratio.transformer import WaistToHeightRatio


# pylint: disable=too-few-public-methods
class ProcessingService:
    """Encapsulates preprocessing pipeline configuration."""

    @property
    def pipeline(self) -> Pipeline:
        """Returns the full preprocessing pipeline.

        Steps (in order):
            1. age_binner: bins RIDAGEYR into one-hot age groups
            2. waist_to_height_ratio: computes ratio, drops BMXWAIST and BMXHT
            3. cholesterol_missingness: adds missingness flag for told_high_cholesterol
            4. column_selector: keeps exactly the 13 model features in canonical order
            5. imputer: fills NaNs per column-group strategy (zero/median/passthrough)
            6. scaler: standardises continuous features; binary features passed through

        Returns:
            Pipeline: Configured sklearn Pipeline ready to fit or transform.
        """
        return Pipeline([
            ("age_binner", AgeBinner()),
            ("waist_to_height_ratio", WaistToHeightRatio()),
            ("cholesterol_missingness", CholesterolMissingness()),
            ("column_selector", ColumnSelector()),
            ("imputer", Imputer()),
            ("scaler", Scaler()),
        ])
