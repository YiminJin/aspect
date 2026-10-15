#!/usr/bin/env python3
"""Render the retained 7B profiles; no simulation or history modification."""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bp3-section7b-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analyze_section7b import ROOT, OUTPUT, read, SCALES


with PdfPages(OUTPUT / "profiles.pdf") as pdf:
    for variant in ["creep", "perturbed"]:
        fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True)
        for backend, color in [("fe", "tab:blue"), ("particle", "tab:orange")]:
            directory = ROOT / f"build-bp3-native/runs/7b-{backend}-{variant}"
            for step, style in [(0, ":"), (10, "-")]:
                p = read(directory / f"coupled-profile-{step}.csv")
                series = [("V_m_s", SCALES["V_m_s"], 0., "V / Vp"),
                          ("theta_s", SCALES["theta_s"], 0., "theta / theta0"),
                          ("Omega", 1., 0., "Omega"),
                          ("slip_m", SCALES["slip_m"], 0., "slip / (Vp T)"),
                          ("shear_Pa", 1000., SCALES["shear_Pa"], "shear perturbation [kPa]"),
                          ("raw_normal_Pa", 1000., 50e6, "normal perturbation [kPa]"),
                          ("pressure_Pa", 1000., 0., "pressure contribution [kPa]"),
                          ("deviatoric_Pa", 1000., 0., "tensor contribution [kPa]")]
                for ax, (field, scale, background, label) in zip(axes.flat, series):
                    ax.plot(p["s_m"], (p[field]-background)/scale, style,
                            color=color, linewidth=1.3, label=f"{backend}, step {step}")
                    ax.set_ylabel(label)
                    ax.grid(alpha=.2)
                if step == 10:
                    axes[2, 1].plot(p["s_m"], (p["filtered_normal_Pa"]-50e6)/1000.,
                                    "--", color=color, linewidth=1.3,
                                    label=f"{backend}, filtered, step 10")
        axes[0, 0].legend(fontsize=8)
        axes[2, 1].legend(fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("down-dip distance [m]")
        fig.suptitle(f"7B {variant}: initial kinematics and T = 2e6 s accepted response\n"
                     "Same mesh, Gauss3, stationary particles, physical loading, filter20; no refinement reference",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .95))
        pdf.savefig(fig)
        plt.close(fig)
