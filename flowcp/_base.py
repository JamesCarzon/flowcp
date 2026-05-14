from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp


class NonconformityScore(ABC):
    """
    Base class for nonconformity scores compatible with flow().

    Subclasses must implement _score() and fit(), then wire up the
    two vmapped class-level attributes:

        score      = jax.vmap(_score, (None, 0, None))
        score_grad = jax.vmap(jax.value_and_grad(_score, argnums=1), (None, 0, None))

    These must be class-level (not instance) attributes so vmap captures
    the unbound function before self is bound.

    Constraints on _score:
      - Accepts a single (y, yhat) pair with no batch dimension.
      - Returns a non-negative scalar.
      - Must be JAX-differentiable w.r.t. yhat (argnums=1).
      - May read instance attributes set during fit(), but must not mutate state.
    """

    @abstractmethod
    def _score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
        """Single-sample nonconformity score. Returns a non-negative scalar."""
        ...

    @abstractmethod
    def fit(self, y: jnp.ndarray, yhat: jnp.ndarray, alpha) -> None:
        """
        Fit on calibration data and compute self.tau.

        y, yhat : (N, ...) matched calibration arrays
        alpha   : scalar or (A,) array of significance levels in [0, 1]

        Must set self.tau (scalar or (A,) array) before returning.
        Typical pattern:
            self.n = y.shape[0]
            k = jnp.ceil((1 - alpha) * (self.n + 1)).astype(int)
            self.tau = jnp.sort(self.batch_score(y, yhat))[k]
        """
        ...

    def batch_score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
        """Paired scores over a calibration batch: (N,...),(N,...) -> (N,)."""
        return jax.vmap(self._score, in_axes=(0, 0))(y, yhat)

    # Subclasses must define these at the class level, e.g.:
    #   score      = jax.vmap(_score, (None, 0, None))
    #   score_grad = jax.vmap(jax.value_and_grad(_score, argnums=1), (None, 0, None))


class LocalizedNonconformityScore(NonconformityScore):
    """
    Extension for scores that support input-conditional (localized) thresholds.

    fit_localizer() and tau_localized() are used only at inference time to
    produce a covariate-dependent tau. flow() itself only calls score_grad
    and self.tau, so the localized threshold must be set on self.tau manually
    before calling flow() if localization is desired.
    """

    @abstractmethod
    def fit_localizer(self, x_cal: jnp.ndarray) -> None:
        """
        Precompute covariate features from calibration inputs x_cal: (N, ...).
        Must be called after fit(). Stores whatever is needed for tau_localized().
        """
        ...

    @abstractmethod
    def tau_localized(self, x_star: jnp.ndarray, *, alpha: float, **kwargs) -> jnp.ndarray:
        """
        Return a localized scalar threshold for a single test covariate x_star.
        Requires fit() and fit_localizer() to have been called first.
        """
        ...
