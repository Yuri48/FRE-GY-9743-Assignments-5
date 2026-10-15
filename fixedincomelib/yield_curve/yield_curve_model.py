"""IFR curve objects and interpolator setup based on the course template."""

import numpy as np
import QuantLib as ql
from typing import List, Optional

from fixedincomelib.date import Date, accrued
from fixedincomelib.data import DataCollection
from fixedincomelib.model import (
    Model, ModelComponent, ModelType, BuildMethodCollection,
    ModelBuilderRegistry, ModelDeserializerRegistry,
)
from fixedincomelib.product import Product
from fixedincomelib.utilities import Interpolator1D, InterpolatorFactory
from fixedincomelib.yield_curve.build_method import YieldCurveIndexBuildMethod, YieldCurveFundingBuildMethod

__all__ = ["YieldCurve", "YieldCurveModelComponent"]


class YieldCurve(Model):

    _version = 1
    _model_type = ModelType.YIELD_CURVE

    def __init__(
        self,
        value_date: Date,
        data_collection: DataCollection,
        build_method_collection: BuildMethodCollection,
    ) -> None:
        super().__init__(value_date, self._model_type, data_collection, build_method_collection)

    def serialize(self) -> dict:
        content = {}
        content["VERSION"] = YieldCurve._version
        content["MODEL_TYPE"] = YieldCurve._model_type.to_string()
        content["VALUE_DATE"] = self.value_date.ISO()
        content["BUILD_METHOD_COLLECTION"] = self.build_method_collection.serialize()
        content["DATA_COLLECTION"] = self.data_collection.serialize()
        return content

    @classmethod
    def deserialize(cls, input_dict: dict) -> "YieldCurve":
        input_dict_ = input_dict.copy()
        assert "VERSION" in input_dict_
        version = input_dict_["VERSION"]
        assert "MODEL_TYPE" in input_dict_
        model_type = input_dict_["MODEL_TYPE"]
        assert "VALUE_DATE" in input_dict_
        value_date = Date(input_dict_["VALUE_DATE"])
        bmc = BuildMethodCollection.deserialize(input_dict_["BUILD_METHOD_COLLECTION"])
        dc = DataCollection.deserialize(input_dict_["DATA_COLLECTION"])
        # find modelbuilder
        func = ModelBuilderRegistry().get(model_type)
        return func(value_date, dc, bmc)

    def discount_factor(
        self, index: ql.Index | str, expiry_date: Date, payment_currency: Optional[str] = None
    ):
        """Combine local discount factors recursively through component references.

        ``payment_currency`` is reserved for future use and does not affect the
        calculation. The reference is read from each stored component identifier.
        """
        visited = set()

        def discount_factor_with_reference(component_index):
            this_component: YieldCurveModelComponent = self.retrieve_model_component(
                component_index
            )
            this_index = this_component.component_identifier
            component_name = this_index.name()
            if component_name in visited:
                raise ValueError(f"Circular yield-curve reference: {component_name}")
            visited.add(component_name)

            df = 1.0
            reference_index = getattr(this_index, "reference_index", None)
            if reference_index is not None:
                df = discount_factor_with_reference(reference_index)
            return df * this_component.discount_factor(expiry_date)

        return discount_factor_with_reference(index)

    def discount_factor_gradient_wrt_state(
        self, index: ql.Index | str, expiry_date: Date,
        gradient_vector: List[np.ndarray],
        scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        """Write raw DF derivatives into component-ordered blocks in place.

        The model discount factor follows the reference chain of ``index``,

            D(T) = D_k(T) * D_{k-1}(T) * ... * D_1(T),

        where D_k is the local factor of the requested component and the
        others belong to its reference, that reference's reference, and so on
        (for ``SOFR-1B-FLAT``: D_funding = D_spread * D_SOFR). By the product
        rule, the derivative with respect to one component's state is its own
        local derivative times the product of every *other* local factor in
        the chain,

            dD/df^(j) = (prod_{i != j} D_i) * dD_j/df^(j),

        and components outside the chain have zero derivative.

        ``gradient_vector`` is a list with one block per model component in
        registration order; it is (re)allocated with ``resize_gradient`` when
        needed. ``scaler`` multiplies the new contribution. With
        ``accumulate=False`` every block is replaced (blocks outside the chain
        become zero); with ``True`` the contribution is added to the existing
        contents. Returns ``None``.
        """
        self.resize_gradient(gradient_vector)
        if not accumulate:
            for block in gradient_vector:
                block.fill(0.0)

        visited = set()

        def gradient_with_reference(component_index, upstream_factor: float) -> float:
            """Accumulate this component's contribution, then its references'.

            ``upstream_factor`` is the product of the local discount factors of
            the components that reference this one (already visited on the way
            down). Returns the discount factor of the sub-chain that starts
            here: this component's local factor times everything it references.
            """
            this_component: YieldCurveModelComponent = self.retrieve_model_component(
                component_index
            )
            this_index = this_component.component_identifier
            component_name = this_index.name()
            if component_name in visited:
                raise ValueError(f"Circular yield-curve reference: {component_name}")
            visited.add(component_name)

            local_df = this_component.discount_factor(expiry_date)
            downstream_factor = 1.0
            reference_index = getattr(this_index, "reference_index", None)
            if reference_index is not None:
                downstream_factor = gradient_with_reference(
                    reference_index, upstream_factor * local_df
                )

            # Product rule: this component's local derivative, scaled by the
            # local factors of everything else in the chain (above and below).
            block = self.retrieve_model_component_index(component_name)
            this_component.discount_factor_gradient_wrt_state(
                expiry_date,
                gradient_vector[block],
                scaler=scaler * upstream_factor * downstream_factor,
                accumulate=True,
            )
            return local_df * downstream_factor

        gradient_with_reference(index, 1.0)


class YieldCurveModelComponent(ModelComponent):

    def __init__(
        self,
        value_date: Date,
        component_identifier: ql.Index,
        state_data: np.ndarray,
        build_method: YieldCurveIndexBuildMethod | YieldCurveFundingBuildMethod,
        calibration_product: Optional[List[Product]] = None,
        calibration_funding: Optional[List[str]] = None,
        market_data: Optional[List] = None,
    ) -> None:

        super().__init__(
            value_date,
            component_identifier,
            state_data,
            build_method,
            [] if calibration_product is None else calibration_product,
            [] if calibration_funding is None else calibration_funding,
            [] if market_data is None else market_data,
        )
        assert len(state_data) == 2
        self.num_state_data_ = len(state_data[0])
        self.interpolator_ = InterpolatorFactory.create_1d_interpolator(
            state_data[0],
            state_data[1],
            self.build_method.interpolation_method,
            self.build_method.extrapolation_method,
        )

    def perturb_model_parameter(
        self, parameter_id: int, perturb_size: float, override_parameter: Optional[bool] = False
    ):
        super().perturb_model_parameter(parameter_id, perturb_size, override_parameter)
        self.interpolator_ = InterpolatorFactory.create_1d_interpolator(
            self.state_data[0],
            self.state_data[1],
            self.build_method.interpolation_method,
            self.build_method.extrapolation_method,
        )

    @property
    def state_data_interpolator(self) -> Interpolator1D:
        return self.interpolator_

    @property
    def num_state_data(self) -> int:
        return self.num_state_data_

    def discount_factor(self, expiry_date: Date):
        """Discount using this component's IFR only, excluding any reference curve."""
        if expiry_date.serialNumber() == 0:
            raise ValueError("A valid discount-factor date is required.")
        if expiry_date < self.value_date:
            raise ValueError("Discount-factor date cannot precede the model value date.")

        # Use the same time convention as the IFR node construction.
        time_to_expiry = accrued(self.value_date, expiry_date)
        exponent = self.state_data_interpolator.integrate(0.0, time_to_expiry)
        return np.exp(-exponent)

    def discount_factor_gradient_wrt_state(
        self, expiry_date: Date, gradient_vector: np.ndarray,
        scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        """Write raw local DF derivatives into the supplied array in place.

        The local discount factor is D(T) = exp(-I(T)) with
        I(T) = int_0^T f(s) ds and f the IFR interpolated from the node
        values f_i. The integral is linear in the node values, so its
        gradient w_i = dI/df_i comes straight from the interpolator
        (``gradient_of_integrated_value_wrt_ordinate``), and the chain rule
        gives

            dD/df_i = -D(T) * w_i.

        ``scaler`` multiplies the new contribution. With ``accumulate=False``
        the contribution replaces the array's contents; with ``True`` it is
        added to them. The array is modified in place and ``None`` returned.
        """
        if not isinstance(gradient_vector, np.ndarray) or gradient_vector.ndim != 1:
            raise TypeError("gradient_vector must be a one-dimensional NumPy array")
        if not np.issubdtype(gradient_vector.dtype, np.floating):
            raise TypeError("gradient_vector must have a floating-point dtype")
        if len(gradient_vector) != self.num_state_data:
            raise ValueError(
                f"gradient_vector has length {len(gradient_vector)}; "
                f"this component has {self.num_state_data} state variables"
            )

        # discount_factor validates the date and fixes the time convention.
        discount_factor = self.discount_factor(expiry_date)
        time_to_expiry = accrued(self.value_date, expiry_date)
        weights = self.state_data_interpolator.gradient_of_integrated_value_wrt_ordinate(
            0.0, time_to_expiry
        )
        contribution = -scaler * discount_factor * np.asarray(weights, dtype=float)

        if accumulate:
            gradient_vector += contribution
        else:
            gradient_vector[:] = contribution


ModelDeserializerRegistry().register(YieldCurve._model_type.to_string(), YieldCurve.deserialize)
