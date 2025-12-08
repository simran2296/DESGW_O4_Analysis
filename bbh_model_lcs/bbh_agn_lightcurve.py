import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.cosmology import Planck18 as cosmo
from synphot import SourceSpectrum, SpectralElement, Observation
from synphot.models import Empirical1D
from astropy.cosmology import z_at_value

def t_ram(M_BBH, kick_velocity):
    return 20 * (M_BBH / 100.0) * (kick_velocity / 200.0) ** (-3)

def t_diff(M_SMBH, aspect_ratio, height, optical_depth):
    return (
        8
        * (optical_depth / 100.0)
        * ( height / (aspect_ratio * 0.01))
        * (aspect_ratio / 700.0)
        * (M_SMBH / 1e8)
    )

def t_end(M_SMBH, kick_velocity, a, aspect_ratio, theta):
    return (
        67
        * (kick_velocity / 200.0) ** (-1)
        * (a / 700.0)
        * (M_SMBH / 1e8)
        * (aspect_ratio / 0.01)
        * (1.0 / np.sin(theta / 60.0))
    )

def L_BHL(M_BBH, kick_velocity, density, radiative_efficiency=0.1):
    return (
        2.5 * 10**45
        * (radiative_efficiency / 0.1)
        * (M_BBH / 100.0) ** 2
        * ((kick_velocity + 50.0) / 200.0) ** (-3)
        * (density / 10**(-10))
    ) * u.erg / u.s

def t_exit(height, mp_optical_depth, kick_velocity):
    return (height * np.sqrt(2 * np.log(mp_optical_depth))) / kick_velocity

def gaussian_rise(rise_array, peak_lumi):
    t = rise_array
    t0 = t[-1]
    tg = t0 - t[0]
    # avoid division by zero
    if tg == 0:
        return np.zeros_like(t) + peak_lumi
    start_val = np.exp(-((t[0] - t0)**2) / (2 * tg**2))
    return peak_lumi * (np.exp(-((t - t0)**2) / (2 * tg**2)) - start_val) / (1 - start_val)

def exponential_decay(fall_array, peak_lumi):
    t = fall_array
    t0 = t[0]
    t_end = t[-1]
    te = t_end - t0
    if te == 0:
        return np.zeros_like(t) + peak_lumi
    return peak_lumi * (np.exp(-(t - t0) / te) - np.exp(-(t_end - t0) / te)) / (1 - np.exp(-(t_end - t0) / te))

# runnoe-based L_lambda_from_Lbol 
_runnoe_coeffs = {
    1450: {'C': 4.2},
    3000: {'C': 5.2},
    5100: {'C': 8.1},
}

def L_lambda_from_Lbol(Lbol, lam_eff):
    lam_value = lam_eff if not hasattr(lam_eff, 'to') else lam_eff.to(u.AA).value
    ref_wavelength = min(_runnoe_coeffs.keys(), key=lambda w: abs(w - lam_value))
    C = _runnoe_coeffs[ref_wavelength]['C']
    return Lbol / C

class BBHAGNLightcurve:
    """
    BBH-AGN lightcurve generator class.
    - events: dictionary of the bbh events with their parameters
    - event_name: eg. "S250328ae"
    """

    def __init__(self, events, event_name, filter_file="data/decam_transmission_curves.txt",
                 n_wav=2000, wav_min=3000, wav_max=11000,
                 radiative_efficiency=0.1, aspect_ratio=0.01, a_rg=700, optical_depth=100, mp_optical_depth=100):
        self.events = events
        self.event_name = event_name
        if event_name not in events:
            raise ValueError(f"Event '{event_name}' not found in events dict")
        self.event = events[event_name]
        self.filter_file = filter_file
        self.n_wav = n_wav
        self.wav_grid = np.linspace(wav_min, wav_max, n_wav) * u.AA
        self.radiative_efficiency = radiative_efficiency
        self.aspect_ratio = aspect_ratio
        self.a_rg = a_rg
        self.optical_depth = optical_depth
        self.mp_optical_depth = mp_optical_depth

        self._load_filters()

        self.dL_Mpc = float(self.event.get("dL", 0.0))
        if self.dL_Mpc <= 0:
            raise ValueError("Event dL must be positive (in Mpc)")
        self.dL = (self.dL_Mpc * u.Mpc).to(u.cm)
        self.z = float(z_at_value(cosmo.luminosity_distance, self.dL))

    def _load_filters(self):
        data = np.loadtxt(self.filter_file, comments="#")
        lam = data[:, 0] * u.AA
        g = data[:, 1]
        r = data[:, 2]
        i = data[:, 3]
        z = data[:, 4]

        sort_idx = np.argsort(lam.value)
        lam = lam[sort_idx]
        g = g[sort_idx]
        r = r[sort_idx]
        i = i[sort_idx]
        z = z[sort_idx]

        self.filters = {
            "g": SpectralElement(Empirical1D(points=lam.value, lookup_table=g), wave_unit=lam.unit),
            "r": SpectralElement(Empirical1D(points=lam.value, lookup_table=r), wave_unit=lam.unit),
            "i": SpectralElement(Empirical1D(points=lam.value, lookup_table=i), wave_unit=lam.unit),
            "z": SpectralElement(Empirical1D(points=lam.value, lookup_table=z), wave_unit=lam.unit),
        }

        self.eff_wavelengths = {}
        for band, bp in self.filters.items():
            points = bp.model.points
            trans = bp.model.lookup_table
            lam_vals = points * u.AA
            eff = (np.trapz(lam_vals.value * trans, lam_vals.value) / np.trapz(trans, lam_vals.value))
            self.eff_wavelengths[band] = eff

    def compute_timescales(self, M_BBH, kick_velocity, height, theta, mass_SMBH):
        t_start = t_ram(M_BBH, kick_velocity)  # days
        diff = t_diff(mass_SMBH, self.a_rg, height, self.optical_depth)

        t_peak = 2.0 * t_start  
        t_end_val = t_end(mass_SMBH, kick_velocity, self.a_rg, self.aspect_ratio, theta)  # days
        
        #print(f"t_end={t_end_val:.2f}")

        return float(t_start), float(t_peak), float(t_end_val)

    def generate_lightcurve(self, M_BBH, kick_velocity, density, height, theta, mass_SMBH,
                            n_rise=100, n_fall=100, radiative_efficiency=None):
        """
        Build the bolometric luminosity lightcurve and return times (days) and luminosities (astropy Quantity erg/s).
        Inputs:
          - M_BBH: BBH mass in SOLAR MASSES
          - kick_velocity: km/s
          - density: g/cm^3 (same units you used)
          - height: same units used in t_diff/t_end formulas (unitless fraction of a or absolute consistent with previous usage)
          - theta: degrees (used in t_end)
          - mass_SMBH: SMBH mass in SOLAR MASSES
        """
        if radiative_efficiency is None:
            radiative_efficiency = self.radiative_efficiency

        # compute times (days)
        t_start, t_peak, t_end_val = self.compute_timescales(M_BBH, kick_velocity, height, theta, mass_SMBH)

        # Create time arrays (days)
        rise_times = np.linspace(t_start, t_peak, n_rise)
        fall_times = np.linspace(t_peak, t_end_val, n_fall)
        times = np.concatenate([rise_times, fall_times[1:]])  # avoid repeating peak

        # compute peak luminosity (vector scalar)
        L_peak = L_BHL(M_BBH, kick_velocity, density, radiative_efficiency)  # Quantity erg/s

        # For each time point produce luminosity using your rise/fall shapes:
        rise_lumis = gaussian_rise(rise_times, L_peak.value)  # returns numpy array (same units as L_peak.value)
        fall_lumis = exponential_decay(fall_times, L_peak.value)

        lumi_vals = np.concatenate([rise_lumis, fall_lumis[1:]]) * u.erg / u.s

        # store attributes
        self.times = times  # days (float array)
        self.luminosities = lumi_vals  # Quantity array erg/s
        self.t_start = t_start
        self.t_peak = t_peak
        self.t_end = t_end_val
        self.L_peak = L_peak

        return times, lumi_vals

    # SED & magnitude computation

    def compute_mags_qso(self):
        """
        Compute QSO-like mags using Runnoe correction / L_lambda_from_Lbol and AB conversion
        This follows your earlier 'mags_from_Lbol' logic (approximate).
        """
        if not hasattr(self, "luminosities"):
            raise RuntimeError("Run generate_lightcurve(...) first.")

        lam_eff = {b: self.eff_wavelengths[b] for b in self.filters.keys()}  # Angstrom floats
        L_vals = self.luminosities.to(u.erg / u.s).value

        rows = []
        for i, L in enumerate(L_vals):
            row = {"time": self.times[i], "L_bol": L}
            for band, lam in lam_eff.items():
                L_lam = L_lambda_from_Lbol(L * u.erg / u.s, lam)  # returns Quantity erg/s
                # convert to L_nu = L_lambda * (lambda / c) (lambda in Å, c in Å/s)
                c_AA_s = 2.99792458e18  # Å / s
                L_nu = (L_lam.to(u.erg / u.s).value) * (lam / c_AA_s)  # erg/s/Hz (approx)
                # F_nu at Earth
                F_nu = L_nu / (4.0 * np.pi * (self.dL.value ** 2) * (1.0 + self.z))
                # AB mag
                # ensure positive F_nu
                if F_nu <= 0 or not np.isfinite(F_nu):
                    m_AB = np.nan
                else:
                    m_AB = -2.5 * np.log10(F_nu) - 48.6
                row[f"mag_qso_{band}"] = m_AB
            rows.append(row)

        df_qso = pd.DataFrame(rows)
        self.df_qso = df_qso
        return df_qso

    def run(self, M_BBH, kick_velocity, density, height, theta, mass_SMBH,
            n_rise=100, n_fall=100, compute_bb=True, compute_qso=True, plot=False):
        """
        Runs full pipeline: generate lightcurve and compute requested magnitudes.
        Returns a dict with times (days), luminosities (Quantity), and dataframes for mags.
        M_BBH, mass_SMBH in SOLAR MASSES; kick_velocity in km/s; density in g/cm^3; height unitless as used in code.
        """
        times, lum = self.generate_lightcurve(M_BBH, kick_velocity, density, height, theta, mass_SMBH,
                                              n_rise=n_rise, n_fall=n_fall)
        if compute_qso:
            df_qso = self.compute_mags_qso()

        if plot:
            self.plot_lightcurve_and_mags()

        return df_qso

    def plot_lightcurve_and_mags(self, bands=["r", "i", "z"], figsize=(8,6), invert_y=True):
        if not hasattr(self, "luminosities"):
            raise RuntimeError("Run generate_lightcurve(...) first.")
        fig, ax = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={"height_ratios":[1,1]})
        # Top: L_bol
        ax[0].plot(self.times, self.luminosities.value, lw=2)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("Time after merger (days)")
        ax[0].set_ylabel(r"$L_{\rm bol}$ (erg s$^{-1}$)")

        # Bottom: mags
        if hasattr(self, "df_bb"):
            for b in bands:
                col = f"mag_bb_{b}"
                if col in self.df_bb.columns:
                    ax[1].plot(self.df_bb["time"], self.df_bb[col], label=f"{b} (BB)")
        if hasattr(self, "df_qso"):
            for b in bands:
                col = f"mag_qso_{b}"
                if col in self.df_qso.columns:
                    ax[1].plot(self.df_qso["time"], self.df_qso[col], linestyle="--", label=f"{b} (QSO)")

        ax[1].invert_yaxis() if invert_y else None
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Time after merger (days)")
        ax[1].set_ylabel("Apparent AB mag")
        ax[1].legend()
        plt.tight_layout()
        plt.show()

