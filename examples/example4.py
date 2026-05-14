"""
Example 4 - Gaussian LRS with image covariate localization (N, H, W, C)

Setting: pixel-wise regression on synthetic spatial fields.

  x  : (H, W, 1)  input "temperature" field drawn from a spatial GP proxy
  y  : (H, W, 1)  target field = smoothed(x) + heteroskedastic noise
  yhat: (H, W, 1) predicted field = spatial mean pooling of x (a weak baseline)

Localization: fit_localizer extracts per-channel statistics and a pooled grid
from each calibration image (default_x_features), then uses RBF-weighted
quantiles to give a tighter threshold for test images that resemble nearby
calibration images.
"""

import flowcp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

key = jax.random.key(42)
key, k1, k2, k3, k4 = jax.random.split(key, 5)

H, W, C = 16, 16, 1
N_cal = 300

# -----------------------------------------------------------------------
# Synthetic data
# -----------------------------------------------------------------------

def make_fields(key, n):
    """
    Each sample is a spatial field with low-frequency structure.
    We simulate this by drawing a coarse random field and resizing it.
    """
    key_coarse, key_noise = jax.random.split(key)
    coarse = jax.random.normal(key_coarse, (n, 4, 4, C))
    x = jax.vmap(lambda u: jax.image.resize(u, (H, W, C), method="linear"))(coarse)
    noise = jax.random.normal(key_noise, (n, H, W, C))
    # noise amplitude scales with the local field intensity
    amplitude = jnp.abs(x) * 0.3 + 0.1
    y = x * 0.8 + amplitude * noise       # smoothed target + heteroskedastic noise
    return x, y

xcal, ycal   = make_fields(k1, N_cal)
xtest, ytest = make_fields(k2, 1)

# Prediction: global spatial mean of x, broadcast back to (H,W,C)
# This is intentionally weak so scores are non-trivial.
def predict(x):
    return jnp.mean(x, axis=(0, 1), keepdims=True) * jnp.ones_like(x)

ycal_hat  = jax.vmap(predict)(xcal)   # (N_cal, H, W, C)
ytest_hat = jax.vmap(predict)(xtest)  # (1, H, W, C)

# -----------------------------------------------------------------------
# Fit LRS
# -----------------------------------------------------------------------

lrs = flowcp.gaussian_lrs(phi_out_hw=(4, 4))   # pool to 4x4 for 16x16 inputs

alpha = 0.1
lrs.fit(ycal, ycal_hat, alpha)

print(f"global tau (alpha={alpha}): {lrs.tau:.4f}")

# -----------------------------------------------------------------------
# Global conformal samples
# -----------------------------------------------------------------------

samp = flowcp.flow(lrs, ycal, ytest_hat, 200)  # (200, H, W, C)
err  = jnp.max(lrs.score(samp, ytest_hat) - lrs.tau)
print(f"global score error (should be ~0): {err:.6f}")

# -----------------------------------------------------------------------
# Localized threshold
# -----------------------------------------------------------------------

# xcal has shape (N_cal, H, W, C) — default_x_features handles (H,W,C) inputs
lrs.fit_localizer(xcal)

tau_local = lrs.tau_localized(xtest[0], alpha=alpha, h=1.0)
print(f"localized tau (alpha={alpha}): {tau_local:.4f}")

lrs.tau = tau_local
samp_local = flowcp.flow(lrs, ycal, ytest_hat, 200)
err_local  = jnp.max(lrs.score(samp_local, ytest_hat) - tau_local)
print(f"localized score error (should be ~0): {err_local:.6f}")

# -----------------------------------------------------------------------
# Show that localized tau varies with the test image
# -----------------------------------------------------------------------

key, k5 = jax.random.split(key)
xtest_batch, _ = make_fields(k5, 5)

taus = jnp.array([
    lrs.tau_localized(xtest_batch[i], alpha=alpha, h=1.0)
    for i in range(5)
])
print(f"\nlocalized tau across 5 different test images:")
print(taus)
print(f"  min={taus.min():.4f}  max={taus.max():.4f}  (global={lrs.tau:.4f})")
