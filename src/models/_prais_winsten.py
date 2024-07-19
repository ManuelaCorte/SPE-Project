from typing import Generator, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn as sbn
from matplotlib.axes import Axes
from numpy.typing import NDArray
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics import residuals_autocorrelation
from src.utils import Float, Matrix


# Class modified from
# https://www.statsmodels.org/dev/examples/notebooks/generated/linear_regression_diagnostics_plots.html
class LinearRegDiagnostic:
    def __init__(self, results: RegressionResultsWrapper) -> None:
        """
        For a linear regression model, generates following diagnostic plots: residual,qq, scale location and leverage plot

        Args:
            results (RegressionResultsWrapper): The fitted model
        """
        self.results = results

        self.y_true = self.results.model.endog
        self.y_predict = self.results.fittedvalues
        self.xvar = self.results.model.exog
        self.xvar_names = self.results.model.exog_names

        self.residual = np.array(self.results.resid)
        influence = self.results.get_influence()
        self.residual_norm = influence.resid_studentized_internal
        self.leverage = influence.hat_matrix_diag
        self.cooks_distance = influence.cooks_distance[0]
        self.nparams = len(self.results.params)
        self.nresids = len(self.residual_norm)

    def plot(self, title: str) -> None:
        """
        Generates diagnostic plots for the fitted model. The plots are:
        1. Residual vs Fitted
        2. Normal Q-Q
        3. Scale-Location
        4. Residuals vs Leverage
        """
        vif = self.vif_table()
        print(vif.to_string(index=False))

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        ax[0, 0] = self.residual_plot(ax=ax[0, 0])
        ax[0, 1] = self.qq_plot(ax=ax[0, 1])
        ax[1, 0] = self.scale_location_plot(ax=ax[1, 0])
        ax[1, 1] = self.leverage_plot(ax=ax[1, 1])

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_predictions(
        self,
        years: Matrix[Literal["N"], np.str_],
        y_true: Matrix[Literal["N"], Float],
        y_test: Matrix[Literal["N"], Float],
        y_covid: Matrix[Literal["N"], Float],
    ) -> None:
        _, ax = plt.subplots(figsize=(10, 15))
        len_test = len(y_true) - len(y_test) - len(y_covid)
        x_train = np.arange(len(y_true))
        x_test = np.arange(len_test, len_test + len(y_test))
        x_covid = np.arange(len(y_true) - len(y_covid), len(y_true))

        sbn.set_theme(style="darkgrid")
        sbn.lineplot(
            x=x_train,
            y=y_true,
            label="True",
            linewidth=2,
            ax=ax,
        )
        ax.vlines(x_test[0], min(y_true), max(y_true), colors="r", linestyles="dashed")
        sbn.lineplot(x=x_test, y=y_test, label="Predicted", linewidth=2, ax=ax)
        ax.vlines(x_covid[0], min(y_true), max(y_true), colors="g", linestyles="dashed")
        sbn.lineplot(
            x=x_covid, y=y_covid, label="Predicted (COVID)", linewidth=2, ax=ax
        )
        ax.set_title("Predictions", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        # insert 1 tick per year
        ax.set_xticks(range(0, len(years), 12))
        # show only the year
        ax.set_xticklabels(
            [years[i][:4] for i in range(0, len(years), 12)], rotation=45
        )

    def residual_plot(self, ax: Axes) -> Axes:
        """
        Residual vs Fitted Plot. Horizontal red line is an indicator that the residual has a linear pattern
        """
        sns.residplot(
            x=self.y_predict,
            y=self.residual,
            lowess=True,
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "red", "lw": 1, "alpha": 0.8},
            ax=ax,
        )

        # annotations
        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.argsort(residual_abs), 0)
        abs_resid_top_3 = abs_resid[:3]
        for i in abs_resid_top_3:
            ax.annotate(
                i,
                xy=(self.y_predict[i], self.residual[i].item()),
                ha="right",
                color="black",
            )

        ax.set_title("Residuals vs Fitted", fontweight="bold")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        return ax

    def qq_plot(self, ax: Axes) -> Axes:
        """
        Standarized Residual vs Theoretical Quantile plot. Points along the diagonal line
        suggest that the residuals are normally distributed.
        """

        QQ = ProbPlot(self.residual_norm)
        QQ.qqplot(line="45", alpha=0.5, lw=1, ax=ax)

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(self.residual_norm)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for i, x, y in self.__qq_top_resid(
            QQ.theoretical_quantiles, abs_norm_resid_top_3  # type: ignore
        ):
            ax.annotate(str(i), xy=(x, y), ha="right", color="black")

        ax.set_title("Normal Q-Q", fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Standardized Residuals")
        return ax

    def scale_location_plot(self, ax: Axes) -> Axes:
        """
        Sqrt(Standarized Residual) vs Fitted values plot

        Used to check homoscedasticity of the residuals.
        Horizontal line will suggest so.
        """
        residual_norm_abs_sqrt = np.sqrt(np.abs(self.residual_norm))

        ax.scatter(self.y_predict, residual_norm_abs_sqrt, alpha=0.5)
        sns.regplot(
            x=self.y_predict,
            y=residual_norm_abs_sqrt,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={"color": "red", "lw": 1, "alpha": 0.8},
            ax=ax,
        )

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            ax.annotate(
                i,
                xy=(self.y_predict[i], residual_norm_abs_sqrt[i]),
                ha="right",
                color="black",
            )

        ax.set_title("Scale-Location", fontweight="bold")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel(r"$\sqrt{|\mathrm{Standardized\ Residuals}|}$")
        return ax

    def leverage_plot(
        self,
        ax: Axes,
        high_leverage_threshold: bool = False,
        cooks_threshold: Optional[str] = "baseR",
    ) -> Axes:
        """
        Residual vs Leverage plot

        Points falling outside Cook's distance curves are considered observation that can sway the fit (outliers).
        Good to have none outside the curves.
        """
        ax.scatter(self.leverage, self.residual_norm, alpha=0.5)

        sns.regplot(
            x=self.leverage,
            y=self.residual_norm,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={"color": "red", "lw": 1, "alpha": 0.8},
            ax=ax,
        )

        # annotations
        leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(
                i,
                xy=(self.leverage[i], self.residual_norm[i]),
                ha="right",
                color="black",
            )

        factors: list[float] = []
        if cooks_threshold == "baseR" or cooks_threshold is None:
            factors = [1.0, 0.5]
        elif cooks_threshold == "convention":
            factors = [4.0 / self.nresids]
        elif cooks_threshold == "dof":
            factors = [4.0 / (self.nresids - self.nparams)]
        else:
            raise ValueError(
                "threshold_method must be one of the following: 'convention', 'dof', or 'baseR' (default)"
            )
        for i, factor in enumerate(factors):
            label = "Cook's distance" if i == 0 else None
            xtemp, ytemp = self.__cooks_dist_line(factor)
            ax.plot(xtemp, ytemp, label=label, lw=1.25, ls="--", color="red")
            ax.plot(xtemp, np.negative(ytemp), lw=1.25, ls="--", color="red")

        if high_leverage_threshold:
            high_leverage = 2 * self.nparams / self.nresids
            if max(self.leverage) > high_leverage:
                ax.axvline(
                    high_leverage, label="High leverage", ls="-.", color="purple", lw=1
                )

        ax.axhline(0, ls="dotted", color="black", lw=1.25)
        ax.set_xlim(0, max(self.leverage) + 0.01)
        ax.set_ylim(min(self.residual_norm) - 0.1, max(self.residual_norm) + 0.1)
        ax.set_title("Residuals vs Leverage", fontweight="bold")
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Standardized Residuals")
        plt.legend(loc="best")
        return ax

    def vif_table(self) -> pd.DataFrame:
        """
        VIF table

        VIF, the variance inflation factor, is a measure of multicollinearity.
        VIF > 5 for a variable indicates that it is highly collinear with the
        other input variables.
        """
        vif_df = pd.DataFrame()
        vif_df["Features"] = self.xvar_names
        vif_df["VIF Factor"] = [
            variance_inflation_factor(self.xvar, i) for i in range(self.xvar.shape[1])
        ]

        return vif_df.sort_values("VIF Factor").round(2)

    def __cooks_dist_line(
        self, factor: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Helper function for plotting Cook's distance curves
        """

        def formula(x: NDArray[np.float64], factor: float, p: int):
            return np.sqrt(factor * p * (1 - x) / x)

        p = self.nparams
        x: NDArray[np.float64] = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x, factor, p)  # type: ignore
        return x, y

    def __qq_top_resid(
        self, quantiles: NDArray[np.float64], top_residual_indices: NDArray[np.int64]
    ) -> Generator[tuple[int, float, float], None, None]:
        """
        Helper generator function yielding the index and coordinates
        """
        offset = 0
        quant_index = 0
        previous_is_negative = None
        for resid_index in top_residual_indices:
            y = self.residual_norm[resid_index]
            is_negative = y < 0
            if previous_is_negative is None or previous_is_negative == is_negative:
                offset += 1
            else:
                quant_index -= offset
            x = (
                quantiles[quant_index]
                if is_negative
                else np.flip(quantiles, 0)[quant_index]
            )
            quant_index += 1
            previous_is_negative = is_negative
            yield resid_index, x, y


class PraisWinstenRegression:
    def __init__(
        self,
        x: Matrix[Literal["N M"], Float],
        y: Matrix[Literal["N"], Float],
        tolerance: float = 1e-3,
    ):
        self.x = x
        self.y = y
        self.tolerance = tolerance

        self._model: Optional[RegressionResultsWrapper] = None
        self._rho: Optional[float] = None
        self.diagnostic: Optional[LinearRegDiagnostic] = None

    @property
    def model(self) -> RegressionResultsWrapper:
        if self._model is None:
            raise ValueError("Model has not been fitted yet.")
        return self._model

    @property
    def rho(self) -> float:
        if self._rho is None:
            raise ValueError("Model has not been fitted yet.")
        return self._rho

    def fit(self) -> None:
        """Fit a regression model using the Prais-Winsten estimation method. This method
        is used to correct for autocorrelation in the residuals of the model by estimating
        it as a first-order autoregressive process.

        Parameters:
            tolerance: The tolerance level used for the Durbin-Watson statistic. The model
            will be refitted until the statistic is within the range [2 - tolerance, 2 + tolerance].
        """
        uncorrected_model = OLS(self.y, self.x).fit()
        rho = self._compute_rho(uncorrected_model)
        model = self._prais_winsten(uncorrected_model, rho)

        dw = residuals_autocorrelation(model.resid, 1)[2].statistic
        while dw < 2 - self.tolerance or dw > 2 + self.tolerance:
            model = self._prais_winsten(model, rho)
            rho = self._compute_rho(model)
            dw = residuals_autocorrelation(model.resid, 1)[
                2
            ].statistic  # durbin-watson statistic
            print("Rho = ", rho)

        self._model = model
        self._rho = rho

    def predict(
        self,
        years: Matrix[Literal["N"], np.str_],
        x_test: Matrix[Literal["N M"], Float],
        y_test: Matrix[Literal["N"], Float],
        x_covid: Matrix[Literal["N M"], Float],
        y_covid: Matrix[Literal["N"], Float],
    ) -> None:
        predicted_test = np.stack(
            [
                self.predict_single(x, x_test[i - 1], y_test[i - 1])
                for i, x in enumerate(x_test)
                if i > 0
            ]
        )
        predicted_test = np.append(
            self.predict_single(x_test[0], None, None), predicted_test
        )

        predicted_covid = np.stack(
            [
                self.predict_single(x, x_covid[i - 1], y_covid[i - 1])
                for i, x in enumerate(x_covid)
                if i > 0
            ]
        )
        predicted_covid = np.append(
            self.predict_single(x_covid[0], None, None), predicted_covid
        )

        true_x = np.concatenate([self.x, x_test, x_covid])
        true_y = np.concatenate([self.y, y_test, y_covid])

        predicted_y = np.concatenate([self.y, predicted_test, predicted_covid])

        if self.diagnostic is None:
            self.diagnostic = LinearRegDiagnostic(self.model)
        print("true_x: ", true_x.shape)
        print("true_y: ", true_y.shape)
        print("predicted_y: ", predicted_y.shape)
        self.diagnostic.plot_predictions(years, true_y, predicted_test, predicted_covid)

    def predict_single(
        self,
        x_t: Matrix[Literal["3 1"], Float],
        x_t1: Optional[Matrix[Literal["3 1"], Float]],
        y_t1: Optional[Matrix[Literal["3 1"], Float]],
    ) -> Matrix[Literal["3 1"], Float]:
        alpha, beta = self.model.params[0], self.model.params[1:]
        if x_t1 is not None and y_t1 is not None:
            y_t = (
                alpha * (1 - self.rho)
                + beta @ (x_t[1:,] - self.rho * x_t1[1:,])
                + self.rho * y_t1
            )
        else:
            y_t = alpha + beta @ x_t[1:,]
        return y_t

    def summary(self) -> str:
        return self.model.summary()

    def plot(self, title: str = "Diagnostic Plots") -> None:
        if self.diagnostic is None:
            self.diagnostic = LinearRegDiagnostic(self.model)

        self.diagnostic.plot(title)

    def _compute_rho(self, model: RegressionResultsWrapper) -> float:
        e_0: Matrix[Literal["N - 1"], Float] = model.resid[1:]
        e_1: Matrix[Literal["N - 1"], Float] = model.resid[:-1]
        rho = np.dot(e_1, e_0) / np.dot(e_1, e_1)
        return rho.item()

    def _prais_winsten(
        self, model: RegressionResultsWrapper, rho: float
    ) -> RegressionResultsWrapper:
        x = model.model.exog
        y = model.model.endog

        # prais winsten transformation for first element
        x_0: Matrix[Literal["1"], Float] = np.sqrt(1 - rho**2) * x[0]
        y_0: Matrix[Literal["1"], Float] = np.sqrt(1 - rho**2) * y[0]

        # cochran orcutt transformation for the rest of the elements
        x_t: Matrix[Literal["N - 1"], Float] = x[1:,] - rho * x[:-1,]
        x_t: Matrix[Literal["N "], Float] = np.append([x_0], x_t, axis=0)
        y_t: Matrix[Literal["N - 1"], Float] = y[1:] - rho * y[:-1]
        y_t: Matrix[Literal["N"], Float] = np.append(y_0, y_t)

        model_ar1 = OLS(y_t, x_t).fit()
        return model_ar1
