# ============================================================
# CDF F(y): part of Eq. (5)
# ============================================================

def empirical_cdf(esp, y):
    """
    esp : shape (n_members,)
        ESP ensemble streamflow forecasts at one time.
    y : scalar
        Possible streamflow value.
    """
    esp = np.asarray(esp)
    return np.mean(esp <= y) # esp <=y, 1, otherwise 0

# ============================================================
# Observation step function F0(y): Eq. (6)
# ============================================================

def observation_cdf(obs, y):
    """
    F0(y) =
        0, y < obs
        1, y >= obs
    """
    return 1.0 if y >= obs else 0.0


def crps_single(esp, obs, num_grid=1000):
    """
    esp : shape (n_members,)
        Ensemble forecast for one basin, one init date, one lead time (LT).
    obs : scalar
        Observed/reference streamflow at target date.
    """

    esp = np.asarray(esp, dtype=float) # Convert predictions to NumPy

    if np.all(np.isnan(esp)) or np.isnan(obs): # Check for missing values
        return np.nan

    y_min = min(np.nanmin(esp), obs)
    y_max = max(np.nanmax(esp), obs)

    if y_min == y_max:
        return 0.0

    ys = np.linspace(y_min, y_max, num_grid) # Divide the streamflow range into 1,000 points
    dy = ys[1] - ys[0]

    integral = 0.0

    for y in ys:
        F_y = empirical_cdf(esp, y)
        F0_y = observation_cdf(obs, y)
        integral += (F_y - F0_y) ** 2

    return integral * dy
