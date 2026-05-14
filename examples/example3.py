"""
Example 3 - Gaussian LRS with localization (N, p)

Demonstrates gaussian_lrs, a negative log likelihood ratio nonconformity score.
The predictive model is N(yhat, diag(sigma^2)) and the null is the marginal
N(mu0, diag(sigma0^2)), both estimated from calibration data.

The second half shows localized thresholds: tau depends on the test covariate
x_star via RBF-weighted quantiles over calibration scores.
"""

import flowcp
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

key = jax.random.key(0)
key, key_xcal, key_xcal_noise, key_xtest, key_noise = jax.random.split(key, 5)

p = 32

# Covariates (used only for localization)
xcal   = jax.random.normal(key_xcal,  (500, p))
xtest  = jax.random.normal(key_xtest, (1,   p))

# Predictions are a noisy linear function of the covariates
W = jax.random.normal(key, (p, p)) * 0.1
ycal_hat  = xcal  @ W
ytest_hat = xtest @ W

# Targets add heteroskedastic noise scaled by the covariate norm
noise_scale = jnp.linalg.norm(xcal, axis=1, keepdims=True) * 0.1 + 0.5
ycal = ycal_hat + jax.random.normal(key_xcal_noise, (500, p)) * noise_scale

#### define score
lrs = flowcp.gaussian_lrs()

#### sample at a fixed alpha level
alpha = 0.1
lrs.fit(ycal, ycal_hat, alpha)

samp = flowcp.flow(lrs, ycal, ytest_hat, 100)

print("fixed alpha score error:", jnp.max(lrs.score(samp, ytest_hat) - lrs.tau))

#### sample across a range of alphas
alpha = jnp.linspace(0.0, 1.0, ycal.shape[0])
lrs.fit(ycal, ycal_hat, alpha)

samp = flowcp.flow(lrs, ycal, ytest_hat, 100)

print("alpha grid score error: ", jnp.max(lrs.score(samp, ytest_hat) - lrs.tau))

#### localized threshold
# fit_localizer precomputes RBF features from calibration covariates
lrs.fit_localizer(xcal)

# tau_localized returns a covariate-dependent threshold for a single test point
tau_local = lrs.tau_localized(xtest[0], alpha=0.1, h=1.0)
print("global tau (alpha=0.1):  ", jnp.sort(lrs.cal_scores)[
    jnp.ceil(0.9 * (lrs.n + 1)).astype(int)
])
print("localized tau (alpha=0.1):", tau_local)

# To use the localized threshold with flow(), set lrs.tau before calling flow()
lrs.tau = tau_local
samp_local = flowcp.flow(lrs, ycal, ytest_hat, 100)
print("localized score error:   ", jnp.max(lrs.score(samp_local, ytest_hat) - tau_local))
