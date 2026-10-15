"""Analytic valuation of a signed fixed-accrual cashflow on an IFR model."""

import numpy as np

from fixedincomelib.model import ModelType
from fixedincomelib.product import ProductFixedAccruedCashflow
from fixedincomelib.valuation import (
    AnalyticValParam, FundingIndexParameter, ValuationEngineProduct,
    ValuationEngineProductRegistry,
)


class ValuationEngineProductFixedAccrued(ValuationEngineProduct):
    """PV and raw first-order IFR risk of one fixed-accrual cashflow.

    The product pays ``C = N * alpha`` on its payment date ``T_p``: ``N`` is
    the signed notional and ``alpha`` the accrual fraction the product has
    already computed. The cashflow is discounted on the funding curve named by
    the ``FUNDING INDEX PARAMETER`` in the valuation parameters, so

        PV      = C * D_funding(T_p)        (future payment)
        dPV/df  = C * dD_funding(T_p)/df    (raw IFR node derivatives)

    A payment on the value date is worth ``C`` and is reported as cash; a
    payment in the past is worth nothing. Neither carries discount-factor
    risk.
    """

    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        if not isinstance(product, ProductFixedAccruedCashflow):
            raise TypeError("product must be a ProductFixedAccruedCashflow")
        if model.model_type != ModelType.YIELD_CURVE.to_string():
            raise TypeError("ValuationEngineProductFixedAccrued requires a YIELD_CURVE model")
        if not valuation_parameters_collection.has_vp_type(FundingIndexParameter._vp_type):
            raise ValueError(
                "A FUNDING INDEX PARAMETER is required to select the discount curve"
            )

        # Select the funding (discount) curve for the product's currency and
        # fail now, rather than at valuation time, if the model lacks it.
        funding_parameter = valuation_parameters_collection.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = funding_parameter.get_funding_index(product.currency)
        self.model.retrieve_model_component(self.funding_index_)

        # Contractual cashflow: signed notional times accrual fraction.
        self.payment_date_ = product.payment_date
        self.cash_amount_ = product.notional * product.accrued

    @property
    def funding_index(self):
        return self.funding_index_

    @property
    def payment_date(self):
        return self.payment_date_

    @property
    def cash_amount(self) -> float:
        return self.cash_amount_

    def calculate_value(self) -> None:
        """Set PV and same-day cash for past, same-day and future payments."""
        if self.payment_date_ < self.value_date:
            # Already paid: nothing left to value.
            self.value_ = 0.0
            self.cash_ = 0.0
        elif self.payment_date_ == self.value_date:
            # Settles today: D = 1, and the amount is cash on the value date.
            self.value_ = self.cash_amount_
            self.cash_ = self.cash_amount_
        else:
            discount_factor = self.model.discount_factor(
                self.funding_index_, self.payment_date_, self.product_.currency.code()
            )
            self.value_ = self.cash_amount_ * discount_factor
            self.cash_ = 0.0

    def calculate_risk(
        self, gradient: list[np.ndarray], scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        """Write ``scaler * C * dD_funding(T_p)/df`` into component-ordered blocks.

        The model's discount-factor gradient already walks the funding curve's
        reference chain, so the product's risk is that gradient scaled by the
        cashflow ``C``. Payments on or before the value date have no
        discount-factor risk: with ``accumulate=False`` every block is zeroed,
        with ``True`` the existing contents are left unchanged.
        """
        self.model.resize_gradient(gradient)
        if self.payment_date_ > self.value_date:
            self.model.discount_factor_gradient_wrt_state(
                self.funding_index_, self.payment_date_, gradient,
                scaler=scaler * self.cash_amount_, accumulate=accumulate,
            )
        elif not accumulate:
            for block in gradient:
                block.fill(0.0)


ValuationEngineProductRegistry().register(
    (ModelType.YIELD_CURVE.to_string(), ProductFixedAccruedCashflow._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductFixedAccrued,
)
