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
        title: Optional[str] = None,
    ) -> None:
        _, ax = plt.subplots(figsize=(15, 20))
        len_test = len(y_true) - len(y_test) - len(y_covid)
        x_train = np.arange(len(y_true))
        x_test = np.arange(len_test, len_test + len(y_test))
        x_covid = np.arange(len(y_true) - len(y_covid), len(y_true))

        sbn.set_theme(style="darkgrid")
        sbn.lineplot(
            x=x_train,
            y=y_true,
            label="True",
            linewidth=1,
            ax=ax,
        )
        ax.vlines(x_test[0], min(y_true), max(y_true), colors="r", linestyles="dashed")
        sbn.lineplot(x=x_test, y=y_test, label="Predicted (Test)", ax=ax)
        ax.vlines(x_covid[0], min(y_true), max(y_true), colors="g", linestyles="dashed")
        sbn.lineplot(
            x=x_covid, y=y_covid, label="Predicted (COVID)", linewidth=2, ax=ax
        )
        if title is not None:
            ax.set_title(title, fontweight="bold")
        else:
            ax.set_title("Predictions", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        # insert 1 tick per year
        ax.set_xticks(range(0, len(years), 12))
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
        x_diff: Optional[Matrix[Literal["N M"], Float]],
        y_diff: Optional[Matrix[Literal["N"], Float]],
        tolerance: float = 0.05,
    ):
        """
        Prais-Winsten regression model for correcting autocorrelation in residuals as a first-order autoregressive process.

        Args:
            x (Matrix[Literal["N M"], Float]): The independent variables
            y (Matrix[Literal["N"], Float]): The dependent variable
            x_diff (Matrix[Literal["N M"], Float]): The independent variables for the first difference (x_t - x_t-1)
            y_diff (Matrix[Literal["N"], Float]): The dependent variable for the first difference (y_t - y_t-1)
            tolerance (float, optional): the tolerance level (p-value) for the Ljung-Box test used for convergence. Defaults to 0.05.
        """

        self.x = x
        self.y = y
        self.x_diff = x_diff
        self.y_diff = y_diff
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
        """
        if self.x_diff is not None and self.y_diff is not None:
            uncorrected_model = OLS(self.y_diff, self.x_diff).fit()
        else:
            uncorrected_model = OLS(self.y, self.x).fit()

        rho = self._compute_rho(uncorrected_model)
        model = self._prais_winsten(uncorrected_model, rho)

        lb = residuals_autocorrelation(model.resid, 1)[0].pvalue
        while self.tolerance > lb:
            model = self._prais_winsten(model, rho)
            rho = self._compute_rho(model)
            lb = residuals_autocorrelation(model.resid, 1)[
                2
            ].statistic  # ljung-box test
            print("Rho = ", rho)

        self._model = model
        self._rho = rho

    def predict(
        self,
        years: Matrix[Literal["N"], np.str_],
        x_test: Matrix[Literal["N M"], Float],
        x_test_diff: Matrix[Literal["N M"], Float],
        y_test: Matrix[Literal["N"], Float],
        y_test_diff: Matrix[Literal["N"], Float],
        x_covid: Matrix[Literal["N M"], Float],
        x_covid_diff: Matrix[Literal["N M"], Float],
        y_covid: Matrix[Literal["N"], Float],
        y_covid_diff: Matrix[Literal["N"], Float],
    ) -> None:
        """Predict the values for the test and COVID data using the fitted model removing the first differences.

        Args:
            years (Matrix[Literal["N"], np.str_]): The years for the data
            x_test (Matrix[Literal["N M"], Float]): The independent variables for the test data
            y_test (Matrix[Literal["N"], Float]): The dependent variable for the test data
            x_covid (Matrix[Literal["N M"], Float]): The independent variables for the COVID data
            y_covid (Matrix[Literal["N"], Float]): The dependent variable for the COVID data
        """
        if self.x_diff is None or self.y_diff is None:
            raise ValueError("Model has not been fitted with first differences.")

        # predicted_test_diff = [0] * len(y_test_diff)
        # ci_diff: list[Matrix[Literal["2"], Float]] = []
        # for i, x in enumerate(x_test_diff):
        #     if i == 0:
        #         y_0, ci = self.predict_single_sample(x, None, None)
        #         predicted_test_diff[0] = y_0
        #         ci_diff.append(ci)
        #     else:
        #         y_t, ci = self.predict_single_sample(
        #             x, x_test_diff[i - 1], y_test_diff[i - 1]
        #         )
        #         predicted_test_diff[i] = y_t
        #         ci_diff.append(ci)
        # predicted_test_diff = np.array(predicted_test_diff)
        # ci_diff = np.stack(ci_diff)

        # predicted_covid_diff = [0] * len(y_covid_diff)
        # ci_diff_covid: list[Matrix[Literal["2"], Float]] = []
        # for i, x in enumerate(x_covid_diff):
        #     if i == 0:
        #         y_0, ci = self.predict_single_sample(x, None, None)
        #         predicted_covid_diff[0] = y_0
        #         ci_diff_covid.append(ci)
        #     else:
        #         y_t, ci = self.predict_single_sample(
        #             x, x_covid_diff[i - 1], y_covid_diff[i - 1]
        #         )
        #         predicted_covid_diff[i] = y_t
        #         ci_diff_covid.append(ci)
        # predicted_covid_diff = np.array(predicted_covid_diff)
        # ci_diff_covid = np.stack(ci_diff_covid)

        # predictions on differences
        predicted_test_diff = np.stack(
            [
                self.predict_single_sample(x, x_test_diff[i - 1], y_test_diff[i - 1])
                for i, x in enumerate(x_test_diff)
                if i > 0
            ]
        )
        predicted_test_diff = np.append(
            self.predict_single_sample(x_test_diff[0], None, None),
            predicted_test_diff,
        )
        predicted_covid_diff = np.stack(
            [
                self.predict_single_sample(x, x_covid_diff[i - 1], y_covid_diff[i - 1])
                for i, x in enumerate(x_covid_diff)
                if i > 0
            ]
        )
        predicted_covid_diff = np.append(
            self.predict_single_sample(x_covid_diff[0], None, None),
            predicted_covid_diff,
        )

        # one step ahead predictions
        predicted_test_one_ahead = np.stack(
            [y_test[i - 1] + predicted_test_diff[i] for i in range(1, len(y_test_diff))]
        )
        np.append(self.y_diff[-1] + predicted_test_diff[0], predicted_test_one_ahead)
        predicted_covid_one_ahead = np.stack(
            [
                y_covid[i - 1] + predicted_covid_diff[i]
                for i in range(1, len(y_covid_diff))
            ]
        )
        np.append(self.y_diff[-1] + predicted_covid_diff[0], predicted_covid_one_ahead)

        # predict based on previous predictions instead of true values
        predicted_test: list[float] = [0] * len(y_test_diff)
        predicted_test[0] = y_test[0] + predicted_test_diff[0]
        for i in range(1, len(y_test_diff)):
            predicted_test[i] = predicted_test[i - 1] + predicted_test_diff[i]

        predicted_covid: list[float] = [0] * len(y_covid_diff)
        predicted_covid[0] = y_covid[0] + predicted_covid_diff[0]
        for i in range(1, len(y_covid_diff)):
            predicted_covid[i] = predicted_covid[i - 1] + predicted_covid_diff[i]

        true_y = np.concatenate([self.y, y_test, y_covid])
        true_y_diff = np.concatenate([self.y_diff, y_test_diff, y_covid_diff])
        if self.diagnostic is None:
            self.diagnostic = LinearRegDiagnostic(self.model)
        self.diagnostic.plot_predictions(
            years,
            true_y_diff,
            predicted_test_diff,
            predicted_covid_diff,
            "Predictions on first order differences",
        )
        self.diagnostic.plot_predictions(
            years,
            true_y,
            predicted_test_one_ahead,
            predicted_covid_one_ahead,
            "One step ahead predictions",
        )
        self.diagnostic.plot_predictions(
            years,
            true_y,
            np.array(predicted_test),
            np.array(predicted_covid),
            "Predictions based on previous predictions",
        )

    def predict_single_sample(
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
            y_t = alpha + beta @ x_t[1:,] + self.model.resid[0]

        # compute confidence interval
        # conf_int = self.model.get_prediction(x_t).conf_int()
        return y_t

    def summary(self) -> str:
        return self.model.summary()

    def diagnostic_plots(self, title: str = "Diagnostic Plots") -> None:
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
