import os
os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"

import numpy as np
from astropy.io import fits
from pyrecon import MultiGridReconstruction
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import PchipInterpolator
import json


# ============================================================
# 0) User settings: input / output / important parameters
# ============================================================

# ---------- Cosmology ----------
COSMO = dict(
    H0=67.36,      # km/s/Mpc
    Om0=0.315192,
    Ob0=0.0493,
)

# ---------- Input ----------
VERSION = 5
INPUT_DIR = "/global/cfs/cdirs/desi/science/td/pv/bgsclustering/Y3/"
PATH_BGS = INPUT_DIR + "BGS_BRIGHT_clustering_forPV_data_Mr_lt_-19.0.fits"
PATH_RANDOM_BGS = INPUT_DIR + "BGS_BRIGHT_clustering_forPV_random_Mr_lt_-19.0.fits"
#PATH_PV = f"/global/cfs/cdirs/desi/science/td/pv/fpgalaxies/Y3/v{VERSION}/FP_clustering_data_v{VERSION}.fits"

# ---------- Output ----------
OUTPUT_DIR = "/global/cfs/cdirs/desi/science/td/pv/blinded/Y3/"
PATH_BLINDED_BGS = OUTPUT_DIR + "redshifts/BGS_BRIGHT_clustering_forPV_data_Mr_lt_-19.0_blind.fits"
#PATH_BLINDED_PV = OUTPUT_DIR + f"FP/FP_clustering_data_v{VERSION}_blind.fits"

# ---------- Reconstruction ----------
RECON_CONFIG = dict(
    f=0.53,
    bias=1.0,
    los=None,
    nmesh=128,
    smoothing_radius=10.0,
    boxsize_pad=1.05,   # 5% margin
)

# ---------- Blinding ----------
PATH_BLIND_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "private", "params.yaml")
with open(PATH_BLIND_PARAMS, encoding="utf-8") as stream:
    blind = json.load(stream)
F_REF, F_SHIFT = blind["F_REF"], blind["F_SHIFT"]

# ---------- z(chi) interpolation ----------
ZMAX_INTERP = 1.2
NZ_INTERP = 8000

# ---------- Constants ----------
C_LIGHT = 299792.458  # km/s


# ============================================================
# 1) Helper functions
# ============================================================

def build_cosmology(cfg):
    return FlatLambdaCDM(
        H0=cfg["H0"] * u.km / u.s / u.Mpc,
        Om0=cfg["Om0"],
        Ob0=cfg["Ob0"],
    )


def load_fits_data(path):
    with fits.open(path) as hdul:
        return hdul[1].data.copy()


def sky_to_cartesian(ra_deg, dec_deg, redshift, cosmo):
    """
    Convert RA/DEC/Z -> Cartesian position in (Mpc/h),
    and LOS unit vector.
    """
    ra = np.deg2rad(ra_deg.astype("f8"))
    dec = np.deg2rad(dec_deg.astype("f8"))
    z = redshift.astype("f8")

    chi = cosmo.comoving_distance(z).value * cosmo.h  # Mpc/h

    x = chi * np.cos(dec) * np.cos(ra)
    y = chi * np.cos(dec) * np.sin(ra)
    z3 = chi * np.sin(dec)

    pos = np.vstack([x, y, z3]).T.astype("f8")
    los = np.vstack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ]).T.astype("f8")

    return chi, pos, los


def make_z_of_chi_mpc_h(cosmo, zmax, nz=8000):
    z_grid = np.linspace(0.0, zmax, nz)
    chi_mpc_h_grid = cosmo.comoving_distance(z_grid).value * cosmo.h

    interp = PchipInterpolator(chi_mpc_h_grid, z_grid, extrapolate=False)

    def z_of_chi(chi_mpc_h):
        chi = np.asarray(chi_mpc_h)
        chi = np.clip(chi, chi_mpc_h_grid[0], chi_mpc_h_grid[-1])
        return interp(chi)

    return z_of_chi


def get_box_from_positions(pos, pad=1.05):
    pos_min = pos.min(axis=0)
    pos_max = pos.max(axis=0)
    boxcenter = 0.5 * (pos_min + pos_max)
    boxsize = (pos_max - pos_min).max() * pad
    return boxsize, boxcenter


def run_reconstruction(data_pos, data_weight, random_pos, random_weight, boxsize, boxcenter, recon_cfg):
    recon = MultiGridReconstruction(
        f=recon_cfg["f"],
        bias=recon_cfg["bias"],
        los=recon_cfg["los"],
        nmesh=recon_cfg["nmesh"],
        boxsize=boxsize,
        boxcenter=boxcenter,
    )
    recon.assign_data(data_pos, data_weight)
    recon.assign_randoms(random_pos, random_weight)
    recon.set_density_contrast(smoothing_radius=recon_cfg["smoothing_radius"])
    recon.run()
    return recon


def write_updated_fits(input_path, output_path, updates):
    """
    updates: dict like {"Z": new_z, "PV": new_pv, ...}
    """
    with fits.open(input_path, mode="readonly", memmap=True) as hdul:
        hdu = hdul[1]
        data = hdu.data
        new_data = data.copy()

        for key, value in updates.items():
            new_data[key] = value

        new_hdu = fits.BinTableHDU(data=new_data, header=hdu.header, name=hdu.name)
        new_hdul = fits.HDUList([hdul[0], new_hdu])
        new_hdul.writeto(output_path, overwrite=True)


def compute_alpha(cosmo, z):
    """
    alpha = [ (1+z)c / (chi(z) H(z)) - 1 ] / [ln(10) c]
    with chi in Mpc and H in km/s/Mpc
    """
    chi_mpc = cosmo.comoving_distance(z).value
    Hz = cosmo.H(z).value
    alpha = ((1.0 + z) * C_LIGHT / (chi_mpc * Hz) - 1.0) / (np.log(10.0) * C_LIGHT)
    return alpha


# ============================================================
# 2) Main
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----- Cosmology -----
    cosmo = build_cosmology(COSMO)

    # ----- Load catalogs -----
    data_bgs = load_fits_data(PATH_BGS)
    data_random = load_fits_data(PATH_RANDOM_BGS)
    #data_pv = load_fits_data(PATH_PV)

    # ----- Convert to Cartesian -----
    chi_bgs, pos_bgs, los_bgs = sky_to_cartesian(
        data_bgs["RA"], data_bgs["DEC"], data_bgs["Z"], cosmo
    )
    #chi_pv, pos_pv, los_pv = sky_to_cartesian(
    #    data_pv["RA"], data_pv["DEC"], data_pv["ZCMB"], cosmo
    #)
    _, pos_random, _ = sky_to_cartesian(
        data_random["RA"], data_random["DEC"], data_random["Z"], cosmo
    )

    weight_bgs = np.asarray(data_bgs["WEIGHT"], dtype="f8")
    weight_random = np.asarray(data_random["WEIGHT"], dtype="f8")

    # ----- Reconstruction box -----
    boxsize, boxcenter = get_box_from_positions(pos_bgs, pad=RECON_CONFIG["boxsize_pad"])
    print("boxsize =", boxsize)
    print("boxcenter =", boxcenter)

    # ----- Run reconstruction -----
    recon = run_reconstruction(
        data_pos=pos_bgs,
        data_weight=weight_bgs,
        random_pos=pos_random,
        random_weight=weight_random,
        boxsize=boxsize,
        boxcenter=boxcenter,
        recon_cfg=RECON_CONFIG,
    )

    # ----- Read displacement field -----
    psi_bgs = recon.read_shifts(pos_bgs, field="disp")   # (N_bgs, 3)
    #psi_pv = recon.read_shifts(pos_pv, field="disp")     # (N_pv, 3)

    psi_los_bgs = np.sum(psi_bgs * los_bgs, axis=1)
    #psi_los_pv = np.sum(psi_pv * los_pv, axis=1)

    # ----- Prepare chi -> z interpolation -----
    z_of_chi = make_z_of_chi_mpc_h(cosmo, zmax=ZMAX_INTERP, nz=NZ_INTERP)

    # ----- Blind f -----
    # Shift chi along LOS displacement
    chi_prime_bgs = chi_bgs - F_REF * psi_los_bgs + F_SHIFT * psi_los_bgs
    #chi_prime_pv = chi_pv - F_REF * psi_los_pv + F_SHIFT * psi_los_pv

    # Convert back to z
    z_prime_bgs = z_of_chi(chi_prime_bgs)
    #z_prime_pv_CMB = z_of_chi(chi_prime_pv)

    # Read 1+Z_SUN
    #ratio_sun = (1 + data_pv["Z"])/(1 + data_pv["ZCMB"])

    #z_prime_pv = (1 + z_prime_pv_CMB) * ratio_sun - 1

    # Update eta and PV
    #eta = data_pv["LOGDIST"].astype("f8")
    #delta_eta = np.log10(chi_prime_pv) - np.log10(chi_pv)
    #eta_prime = eta + delta_eta
    #alpha_prime = compute_alpha(cosmo, z_prime_pv)
    # pv_prime = eta_prime / alpha_prime

    # Output paths
    out_bgs = PATH_BLINDED_BGS
    #out_pv = PATH_BLINDED_PV

    # Write blinded BGS
    write_updated_fits(
        PATH_BGS,
        out_bgs,
        {"Z": z_prime_bgs},
    )

    # Write blinded PV
    #write_updated_fits(
    #    PATH_PV,
    #    out_pv,
    #    {"ZCMB": z_prime_pv_CMB, "LOGDIST": eta_prime, "Z": z_prime_pv},
    #)

    print("Done.")


if __name__ == "__main__":
    main()