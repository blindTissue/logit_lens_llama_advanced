"""
Save attention heatmaps to PDF. Styles mirror the in-app table (app) or matplotlib colormap.
"""
from __future__ import annotations

import math
from typing import List, Literal, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

SaveStyle = Literal["app", "colormap", "colormap_clean"]

VALID_SAVE_STYLES = ("app", "colormap", "colormap_clean")
DEFAULT_SAVE_STYLE: SaveStyle = "app"

VIZ_SAVE_KWARGS = {"format": "pdf", "bbox_inches": "tight"}

# Match frontend: rgba(0, 255, 128, val) on dark UI
_APP_BG = (0.06, 0.09, 0.16)  # ~#0f172a
_APP_GREEN = (0.0, 1.0, 0.5)  # rgb(0, 255, 128)
_APP_EDGE = "#334155"
_APP_TEXT = "#f8fafc"


def normalize_save_style(style: Optional[str]) -> SaveStyle:
    if style in VALID_SAVE_STYLES:
        return style  # type: ignore[return-value]
    return DEFAULT_SAVE_STYLE


def _cell_color_app(val: float) -> tuple[float, float, float]:
    return tuple(_APP_BG[i] + val * (_APP_GREEN[i] - _APP_BG[i]) for i in range(3))


def _annotation_fontsize(n: int, *, grid: bool = False) -> float:
    if n > 40:
        return 3.5
    if n > 28:
        return 4.5
    if n > 18:
        return 5.5
    if n > 12:
        return 6.5
    return 5.0 if grid else 8.0


def _figsize_for_matrix(n: int, *, grid: bool = False) -> tuple[float, float]:
    cell = 0.32 if grid else 0.42
    side = max(5.0, min(28.0, n * cell + 2.5))
    return (side, side * 0.9)


def _truncate_tokens(tokens: List[str], max_len: int = 10) -> List[str]:
    return [t[:max_len] for t in tokens]


def _style_axes_dark(ax: plt.Axes) -> None:
    ax.set_facecolor(_APP_BG)
    for spine in ax.spines.values():
        spine.set_color(_APP_EDGE)
    ax.tick_params(colors=_APP_TEXT, labelsize=8)
    ax.xaxis.label.set_color(_APP_TEXT)
    ax.yaxis.label.set_color(_APP_TEXT)
    ax.title.set_color(_APP_TEXT)


def draw_attention_heatmap(
    ax: plt.Axes,
    data: List[List[float]],
    tokens: List[str],
    *,
    save_style: SaveStyle = DEFAULT_SAVE_STYLE,
    subtitle: Optional[str] = None,
    show_cbar: bool = True,
) -> None:
    """Draw one attention matrix on *ax*."""
    arr = np.asarray(data, dtype=float)
    n = arr.shape[0]
    labels = _truncate_tokens(tokens)
    show_annot = save_style != "colormap_clean"
    fontsize = _annotation_fontsize(n, grid=not show_cbar)

    if save_style == "app":
        _style_axes_dark(ax)
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal")
        ax.invert_yaxis()

        for i in range(n):
            for j in range(n):
                val = float(arr[i, j])
                rect = Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=_cell_color_app(val),
                    edgecolor=_APP_EDGE,
                    linewidth=0.6,
                )
                ax.add_patch(rect)
                if show_annot:
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=fontsize,
                        color="black" if val > 0.5 else "white",
                    )

        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Key Token")
        ax.set_ylabel("Query Token")
        if subtitle:
            ax.set_title(subtitle, fontsize=10, pad=8)
        return

    # Colormap styles (matplotlib/seaborn)
    ax.set_facecolor("white")
    sns.heatmap(
        arr,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        ax=ax,
        cbar=show_cbar,
        annot=show_annot,
        fmt=".2f",
        annot_kws={"size": fontsize},
        linewidths=0.3,
        linecolor="#333",
    )
    if subtitle:
        ax.set_title(subtitle, fontsize=10)
    ax.set_xlabel("Key Token")
    ax.set_ylabel("Query Token")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def save_attention_heatmap(
    filepath: str,
    data: List[List[float]],
    tokens: List[str],
    title: str,
    *,
    save_style: SaveStyle = DEFAULT_SAVE_STYLE,
    include_title: bool = True,
) -> None:
    n = len(data)
    fig, ax = plt.subplots(figsize=_figsize_for_matrix(n))
    fig.patch.set_facecolor(_APP_BG if save_style == "app" else "white")
    draw_attention_heatmap(
        ax,
        data,
        tokens,
        save_style=save_style,
        subtitle=title if include_title else None,
    )
    plt.tight_layout()
    plt.savefig(filepath, **VIZ_SAVE_KWARGS)
    plt.close(fig)


def save_attention_grid(
    filepath: str,
    grid_data: List[List[List[float]]],
    tokens: List[str],
    title: str,
    *,
    grid_type: str,
    save_style: SaveStyle = DEFAULT_SAVE_STYLE,
    include_title: bool = True,
) -> None:
    num_plots = len(grid_data)
    cols = 4
    rows = math.ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.8))
    fig.patch.set_facecolor(_APP_BG if save_style == "app" else "white")
    if include_title and title:
        fig.suptitle(title, fontsize=12, color=_APP_TEXT if save_style == "app" else "black", y=1.02)

    axes_flat = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes_flat):
        if i < num_plots:
            sub = None
            if include_title:
                sub = f"Layer {i}" if grid_type == "layer_grid" else f"Head {i}"
            draw_attention_heatmap(
                ax,
                grid_data[i],
                tokens,
                save_style=save_style,
                subtitle=sub,
                show_cbar=False,
            )
            if save_style == "app":
                ax.tick_params(labelsize=6)
        else:
            ax.axis("off")
            if save_style == "app":
                ax.set_facecolor(_APP_BG)

    plt.tight_layout()
    plt.savefig(filepath, **VIZ_SAVE_KWARGS)
    plt.close(fig)
