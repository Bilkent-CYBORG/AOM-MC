import os

os.environ["PATH"] = '/home/sepehr/texlive/2020/bin/x86_64-linux:' + os.environ["PATH"]

import matplotlib.pyplot as plt
import seaborn as sns


def init_plot_params():
    # taken from https://jwalton.info/Embed-Publication-Matplotlib-Latex
    USE_LATEX = True

    sns.set_theme(style='whitegrid')

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": USE_LATEX,
        "font.serif": 'Times New Roman',
        # # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 16,
        "font.size": 16,
        # # Make the legend/label fonts a little smaller
        "legend.fontsize": 11,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16
    }
    plt.rcParams.update(tex_fonts)