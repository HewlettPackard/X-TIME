import os
import tempfile

import numpy as np
import treelite
import treelite.sklearn
from numpy.typing import NDArray

from . import _xtimec as rust  # type: ignore

FloatArray = NDArray[np.float64]


class XTimeModel:
    """A tree-based machine learning model mapped to an ACAM matrix for inference."""

    _model: FloatArray
    _leaf_vector_size: int

    def __init__(self, model: FloatArray, leaf_vector_size: int = 1) -> None:
        self._model = model
        self._leaf_vector_size = leaf_vector_size

    @property
    def raw_model(self) -> FloatArray:
        """
        The raw compiler output, contained within a single NumPy matrix.
        Every row is encoded as:
        [ACAM threshold pairs..., Leaf values... Class ID, Tree ID].
        """
        return self._model

    @property
    def cam(self) -> FloatArray:
        """The ACAM matrix."""
        return self._model[:, : -self._leaf_vector_size - 2]

    @property
    def leaves(self) -> FloatArray:
        """
        The corresponding leaf values for each row of the ACAM matrix.
        CatBoost models may have several leaf values per row,
        while all other models always have a single value.
        """
        return self._model[:, -self._leaf_vector_size - 2 : -2]

    @property
    def class_ids(self) -> FloatArray:
        """The corresponding class ID for each CAM row."""
        return self._model[:, -2]

    @property
    def tree_ids(self) -> FloatArray:
        """The corresponding tree ID for each CAM row."""
        return self._model[:, -1]

    @staticmethod
    def from_catboost(model) -> "XTimeModel":
        """
        Construct an XTimeModel from a CatBoost model.

        Parameters
        ----------
        model : [catboost.CatBoost]
            The CatBoost model.

        Returns
        -------
        [XTimeModel]
            The compiled model.
        """
        assert model.classes_ is not None
        num_class = len(model.classes_)

        tmp = tempfile.mktemp(".json")
        try:
            model.save_model(tmp, format="json", export_parameters=None)
            result = rust.compile_catboost(tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        return XTimeModel(result, num_class)

    @staticmethod
    def from_treelite(model: treelite.Model) -> "XTimeModel":
        """
        Construct an XTimeModel from a Treelite model.

        Parameters
        ----------
        model : [treelite.Model]
            The Treelite model.

        Returns
        -------
        [XTimeModel]
            The compiled model.
        """
        return XTimeModel(rust.compile_treelite(model.dump_as_json(pretty_print=False)))

    @staticmethod
    def from_lightgbm(booster) -> "XTimeModel":
        """
        Construct an XTimeModel from a LightGBM booster.

        Parameters
        ----------
        booster : [lightgbm.Booster]
            The LightGBM booster.

        Returns
        -------
        [XTimeModel]
            The compiled model.
        """
        return XTimeModel.from_treelite(treelite.Model.from_lightgbm(booster))

    @staticmethod
    def from_xgboost(booster) -> "XTimeModel":
        """
        Construct an XTimeModel from an XGBoost booster.

        Parameters
        ----------
        booster : [xgboost.Booster]
            The XGBoost booster.

        Returns
        -------
        [XTimeModel]
            The compiled model.
        """
        return XTimeModel.from_treelite(treelite.Model.from_xgboost(booster))

    @staticmethod
    def from_sklearn(model) -> "XTimeModel":
        """
        Construct an XTimeModel from a Scikit-Learn model.

        Parameters
        ----------
        model : [Scikit-Learn model, e.g. sklearn.ensemble.RandomForestClassifier]
            The Scikit-Learn model.

        Returns
        -------
        [XTimeModel]
            The compiled model.
        """
        return XTimeModel.from_treelite(treelite.sklearn.import_model(model))
