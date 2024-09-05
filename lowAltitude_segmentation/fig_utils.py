from __future__ import annotations

import os
from pathlib import Path


def papermode(plt, size: int | None = None, has_latex: bool = True):
    if has_latex:
        plt.rc("font", family="serif", serif="Times")
        plt.rc("text", usetex=True)
    if size is not None:
        plt.rc("xtick", labelsize=size)
        plt.rc("ytick", labelsize=size)
        plt.rc("axes", labelsize=size)
        plt.rc("figure", labelsize=size)
        plt.rc("legend", fontsize=size)
