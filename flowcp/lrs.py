"""
Negative log likelihood ratio nonconformity scores.

The score is:

    S(y, yhat) = -log p_hat(y | yhat) + log p_0(y)
               = NLL_model(y | yhat) - NLL_null(y)

A positive score means the predictive model explains y worse than the null;
a score near zero means the prediction is essentially as good as the null.

The null model p_0 is the marginal estimated from calibration targets,
and the predictive model p_hat(y | yhat) is centered on yhat with the
same covariance structure. Both use a diagonal Gaussian by default.

For localization, an RBF kernel on covariate features is used to compute
weighted quantiles of calibration scores, following the same pattern as
local_nonconf in local_score.py.
"""

import jax
import jax.numpy as jnp

from .local_score import (
    default_x_features,
    fit_phi_standardizer,
    rbf_weights,
    weighted_quantile,
)
from ._base import LocalizedNonconformityScore


class gaussian_lrs(LocalizedNonconformityScore):
    """
    Negative log likelihood ratio score under diagonal Gaussian models.

    Predictive model : p_hat(y | yhat) = N(yhat, diag(sigma^2))
    Null model       : p_0(y)          = N(mu_0, diag(sigma_0^2))

    Both sigma and (mu_0, sigma_0) are estimated from calibration residuals
    and calibration targets respectively.

    score(y, yhat) = 0.5 * sum_d [ (y_d - yhat_d)^2 / sigma_d^2
                                  - (y_d - mu0_d)^2  / sigma0_d^2 ]

    The log-determinant terms cancel when sigma == sigma_0 (they are reported
    separately here so they do not cancel, which is the correct LR).

    Parameters
    ----------
    eps : float
        Small constant added to estimated variances for numerical stability.
    phi_x_fn : callable, optional
        Feature extractor for covariate localization. Signature:
        (x: ndarray, **kwargs) -> 1-D feature vector.
        Defaults to the pooling-based extractor from local_score.py.
    phi_out_hw : tuple[int, int]
        Spatial pooling grid size passed to the default phi_x_fn.
    phi_eps : float
        Epsilon passed to phi_x_fn for numerical stability.
    """

    def __init__(
        self,
        *,
        eps: float = 1e-6,
        phi_x_fn=default_x_features,
        phi_out_hw=(8, 8),
        phi_eps: float = 1e-6,
    ):
        self.eps = float(eps)
        self.phi_x_fn = phi_x_fn
        self.phi_out_hw = tuple(phi_out_hw)
        self.phi_eps = float(phi_eps)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, y: jnp.ndarray, yhat: jnp.ndarray, alpha) -> None:
        """
        Estimate Gaussian parameters from calibration data and compute tau.

        y, yhat : (N, ...) — matched calibration arrays (arbitrary shape)
        alpha   : scalar or (A,) array in [0, 1]
        """
        self.alpha = alpha
        self.n = y.shape[0]
        self.k_alpha = jnp.ceil((1.0 - self.alpha) * (self.n + 1)).astype(int)

        resid = y - yhat  # (N, ...)

        # Predictive model: center on yhat, variance from residuals
        self.sigma = jnp.std(resid, axis=0) + self.eps  # (...)

        # Null model: marginal of y
        self.mu0    = jnp.mean(y, axis=0)               # (...)
        self.sigma0 = jnp.std(y, axis=0) + self.eps     # (...)

        self.cal_scores = self.batch_score(y, yhat)           # (N,)
        self.tau = jnp.sort(self.cal_scores)[self.k_alpha]    # scalar or (A,)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
        """
        NLLR for a single (y, yhat) pair.  y and yhat have no batch dim.
        """
        var   = self.sigma  ** 2
        var0  = self.sigma0 ** 2

        nll_model = 0.5 * jnp.sum((y - yhat) ** 2 / var   + jnp.log(var))
        nll_null  = 0.5 * jnp.sum((y - self.mu0) ** 2 / var0 + jnp.log(var0))

        return nll_model - nll_null

    # vmap over the y-batch dimension; yhat and self are broadcast
    score      = jax.vmap(_score, (None, 0, None))
    score_grad = jax.vmap(jax.value_and_grad(_score, argnums=1), (None, 0, None))

    # ------------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------------

    def fit_localizer(self, x_cal: jnp.ndarray) -> None:
        """
        Precompute RBF-kernel weights from calibration covariates.

        x_cal : (N, ...) covariate array (e.g. input fields or context features)
        Requires fit() to have been called first (to set self.cal_scores).
        """
        Phi = jax.vmap(
            lambda x: self.phi_x_fn(x, out_hw=self.phi_out_hw, eps=self.phi_eps)
        )(x_cal)
        self.Phi_cal = Phi
        self.phi_mu, self.phi_sd = fit_phi_standardizer(Phi)

    def tau_localized(
        self,
        x_star: jnp.ndarray,
        *,
        alpha: float,
        h: float = 1.0,
    ) -> jnp.ndarray:
        """
        Localized threshold for a single test covariate x_star.

        Uses RBF-weighted quantile of calibration scores at level 1 - alpha.
        Requires fit() and fit_localizer() to have been called first.

        x_star : (...) single covariate (no batch dimension)
        alpha  : scalar significance level in [0, 1]
        h      : RBF bandwidth (in standardized feature space)
        """
        phi_star = self.phi_x_fn(x_star, out_hw=self.phi_out_hw, eps=self.phi_eps)
        w = rbf_weights(self.Phi_cal, phi_star, self.phi_mu, self.phi_sd, h=h)
        return weighted_quantile(self.cal_scores, w, q=1.0 - alpha)
