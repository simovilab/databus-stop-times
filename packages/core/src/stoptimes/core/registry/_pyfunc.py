"""The MLflow pyfunc wrapper stored with every artifact the MLflow backend publishes.

stoptimes itself never runs it (it downloads the artifact directory and loads it the normal way, with
the drift, asset and feature checks). It exists so MLflow-native tools (`mlflow.pyfunc.load_model`,
`mlflow models serve`) can use a stoptimes model too. Its input is the **feature frame**, already
computed: live features (schedule, trip state) come from the stoptimes live session, not from here.
"""

from __future__ import annotations

import pickle  # noqa: S403 - loads the registry-owned artifact this wrapper was published with
from pathlib import Path
from typing import Any

try:  # only importable where mlflow is installed; the backend imports this module lazily
    from mlflow.pyfunc import PythonModel
except ImportError:  # pragma: no cover - exercised only without the extra
    PythonModel = object  # type: ignore[assignment,misc]

#: Key of the artifact directory in the pyfunc's `artifacts` mapping.
ARTIFACT_KEY = "stoptimes"


class StoptimesPyfunc(PythonModel):  # type: ignore[misc,valid-type]
    """`predict(feature_frame) -> seconds`, using the artifact's pickled model."""

    def load_context(self, context: Any) -> None:
        import json

        directory = Path(context.artifacts[ARTIFACT_KEY])
        meta = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
        self.features: list[str] = list(meta.get("features", []))
        from stoptimes.core.contracts.artifact import ModelFormat

        model_file = directory / ModelFormat(meta.get("format", ModelFormat.PICKLE.value)).filename
        with open(model_file, "rb") as fh:
            self._model = pickle.load(fh)  # noqa: S301

    def predict(self, context, model_input, params=None):  # noqa: ANN001,ANN201 - mlflow infers a schema from hints
        frame = model_input[self.features] if self.features else model_input
        return self._model.predict(frame)
