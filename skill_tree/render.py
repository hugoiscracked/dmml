"""
Render the DMML model-selection skill tree.

Examples:
  ./env/bin/python skill_tree/render.py --full
  ./env/bin/python skill_tree/render.py --week 6 --highlight-current
  ./env/bin/python skill_tree/render.py --all-weeks --future hide
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path as MplPath


DEFAULT_DATA = HERE / "tree.json"
DEFAULT_OUT = HERE / "out"

BG = "#0d1117"
TEXT_BRIGHT = "#ffffff"
TEXT_DIM = "#8b949e"
LOCKED_TEXT = "#59636e"

COL = {
    "root": "#58a6ff",
    "tab": "#3fb950",
    "ts": "#f78166",
    "unsup": "#2dd4bf",
    "img": "#bc8cff",
    "seq": "#ffa657",
    "shap_special": "#e3b341",
}

BADGE_BG = {
    "sklearn": "#1f6feb",
    "statsmodels": "#2d6a9f",
    "xgboost": "#b08000",
    "lightgbm": "#2e7d32",
    "catboost": "#b85c00",
    "pytorch": "#b93221",
    "huggingface": "#b07300",
    "shap": "#276e27",
}

HW, HH = 3.6, 1.25
MW, MH = 3.0, 2.35
BAR_H = 0.72


def load_tree(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["node_map"] = {n["id"]: n for n in data["nodes"]}
    return data


def branch_col(key: str) -> str:
    return COL.get(key, COL["tab"])


def wrap(text: str, width: int = 28) -> list[str]:
    return textwrap.wrap(text, width=width) if text else []


def node_hh(node: dict) -> float:
    return HH if node.get("kind") == "header" else MH


def unlocked(node: dict, week: int | None) -> bool:
    return week is None or int(node.get("week", 0)) <= week


def current(node: dict, week: int | None) -> bool:
    return week is not None and int(node.get("week", 0)) == week


def visible(node: dict, week: int | None, future: str) -> bool:
    return unlocked(node, week) or future == "dim"


def alpha_for(node: dict, week: int | None, future: str) -> float:
    if unlocked(node, week):
        return 1.0
    return 0.18 if future == "dim" else 0.0


def color_for(node: dict, week: int | None, future: str) -> str:
    if unlocked(node, week):
        return branch_col(node["branch"])
    return "#30363d" if future == "dim" else branch_col(node["branch"])


def text_color_for(node: dict, week: int | None, future: str) -> str:
    if unlocked(node, week):
        return TEXT_BRIGHT
    return LOCKED_TEXT if future == "dim" else TEXT_BRIGHT


def bezier_edge(ax, tree: dict, src_id: str, dst_id: str, week: int | None, future: str):
    nodes = tree["node_map"]
    src = nodes[src_id]
    dst = nodes[dst_id]

    if not (visible(src, week, future) and visible(dst, week, future)):
        return

    is_shap_edge = dst_id == "shap_node"
    color = branch_col("shap_special") if is_shap_edge else color_for(dst, week, future)
    edge_alpha = 0.55 if unlocked(src, week) and unlocked(dst, week) else 0.12

    xs, ys = src["position"]
    xd, yd = dst["position"]
    y0 = ys - node_hh(src)
    y1 = yd + node_hh(dst)
    dy = abs(y0 - y1)
    tension = max(dy * 0.48, 1.8)

    verts = [(xs, y0), (xs, y0 - tension), (xd, y1 + tension), (xd, y1)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    linestyle = (0, (5, 4)) if is_shap_edge else "solid"
    patch = mpatches.PathPatch(
        MplPath(verts, codes),
        fc="none",
        ec=color,
        lw=1.3 if is_shap_edge else 1.4,
        alpha=edge_alpha,
        zorder=1,
        linestyle=linestyle,
        capstyle="round",
    )
    ax.add_patch(patch)

    if not is_shap_edge and edge_alpha > 0.2:
        ax.plot(xd, y1, "o", ms=3, color=color, alpha=0.7, zorder=2)


def draw_week_badge(ax, node: dict, week: int | None, future: str, x: float, y: float):
    if week is None or not visible(node, week, future):
        return
    label = f"W{int(node.get('week', 0)):02d}" if node.get("week", 0) else "START"
    alpha = 0.92 if unlocked(node, week) else 0.20
    ax.text(
        x,
        y,
        label,
        ha="right",
        va="center",
        fontsize=4.8,
        fontweight="bold",
        color=TEXT_DIM if unlocked(node, week) else LOCKED_TEXT,
        alpha=alpha,
        zorder=6,
    )


def header_node(ax, node: dict, week: int | None, future: str, highlight_current: bool):
    if not visible(node, week, future):
        return

    x, y = node["position"]
    col = color_for(node, week, future)
    alpha = alpha_for(node, week, future)
    is_current = current(node, week) and highlight_current

    if is_current:
        ax.add_patch(
            FancyBboxPatch(
                (x - HW - 0.25, y - HH - 0.25),
                2 * (HW + 0.25),
                2 * (HH + 0.25),
                boxstyle="round,pad=0.12",
                lw=11,
                ec=col,
                fc="none",
                alpha=0.28,
                zorder=2,
            )
        )

    ax.add_patch(
        FancyBboxPatch(
            (x - HW, y - HH),
            2 * HW,
            2 * HH,
            boxstyle="round,pad=0.12",
            lw=7,
            ec=col,
            fc="none",
            alpha=0.14 * alpha,
            zorder=2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - HW, y - HH),
            2 * HW,
            2 * HH,
            boxstyle="round,pad=0.12",
            lw=2,
            ec=col,
            fc="#161b22",
            alpha=0.35 + 0.65 * alpha,
            zorder=3,
        )
    )
    ax.text(
        x,
        y,
        node["label"],
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=col if unlocked(node, week) else LOCKED_TEXT,
        alpha=max(alpha, 0.35),
        zorder=4,
        multialignment="center",
        path_effects=[pe.withStroke(linewidth=2, foreground=BG)],
    )
    draw_week_badge(ax, node, week, future, x + HW - 0.25, y - HH + 0.28)


def method_node(ax, node: dict, week: int | None, future: str, highlight_current: bool):
    if not visible(node, week, future):
        return

    x, y = node["position"]
    col = color_for(node, week, future)
    alpha = alpha_for(node, week, future)
    is_shap = node["id"] == "shap_node"
    is_current = current(node, week) and highlight_current
    body_fc = "#1a1608" if is_shap and unlocked(node, week) else "#161b22"

    if is_current:
        ax.add_patch(
            FancyBboxPatch(
                (x - MW - 0.32, y - MH - 0.32),
                2 * (MW + 0.32),
                2 * (MH + 0.32),
                boxstyle="round,pad=0.12",
                lw=11,
                ec=col,
                fc="none",
                alpha=0.25,
                zorder=2,
            )
        )

    if is_shap:
        ax.add_patch(
            FancyBboxPatch(
                (x - MW - 0.30, y - MH - 0.30),
                2 * (MW + 0.30),
                2 * (MH + 0.30),
                boxstyle="round,pad=0.12",
                lw=1.5,
                ec=col,
                fc="none",
                alpha=0.60 * alpha,
                zorder=2,
                linestyle=(0, (5, 3)),
            )
        )

    ax.add_patch(
        FancyBboxPatch(
            (x - MW, y - MH),
            2 * MW,
            2 * MH,
            boxstyle="round,pad=0.1",
            lw=6,
            ec=col,
            fc="none",
            alpha=0.10 * alpha,
            zorder=2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - MW, y - MH),
            2 * MW,
            2 * MH,
            boxstyle="round,pad=0.1",
            lw=1.5,
            ec=col,
            fc=body_fc,
            alpha=0.32 + 0.68 * alpha,
            zorder=3,
        )
    )

    bar_top = y + MH - 0.08
    bar_bottom = bar_top - BAR_H
    ax.add_patch(
        FancyBboxPatch(
            (x - MW + 0.08, bar_bottom),
            2 * MW - 0.16,
            BAR_H,
            boxstyle="round,pad=0.06",
            lw=0,
            fc=col,
            alpha=0.88 * max(alpha, 0.22),
            zorder=4,
        )
    )

    ax.text(
        x,
        (bar_top + bar_bottom) / 2,
        node["label"],
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=text_color_for(node, week, future),
        alpha=max(alpha, 0.45),
        zorder=5,
        multialignment="center",
        path_effects=[pe.withStroke(linewidth=1.5, foreground=col if unlocked(node, week) else BG)],
    )

    cur_y = bar_bottom - 0.13
    use_when = node.get("use_when", "")
    if use_when:
        for line in wrap(f"> {use_when}", width=30)[:2]:
            ax.text(
                x,
                cur_y,
                line,
                ha="center",
                va="top",
                fontsize=6.0,
                color=col if unlocked(node, week) else LOCKED_TEXT,
                alpha=max(alpha, 0.45),
                fontstyle="italic",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=1, foreground=BG)],
            )
            cur_y -= 0.40

    cur_y -= 0.10
    for bullet in node.get("bullets", [])[:3]:
        for line in wrap(f"- {bullet}", width=32)[:2]:
            ax.text(
                x - MW + 0.22,
                cur_y,
                line,
                ha="left",
                va="top",
                fontsize=5.5,
                color=TEXT_DIM if unlocked(node, week) else LOCKED_TEXT,
                alpha=max(alpha, 0.36),
                zorder=5,
                path_effects=[pe.withStroke(linewidth=1, foreground=BG)],
            )
            cur_y -= 0.35
        cur_y -= 0.06

    libs = node.get("libraries", [])
    if libs:
        badge_cy = y - MH + 0.38
        bw = 1.55
        gap = 0.22
        total = len(libs) * bw + (len(libs) - 1) * gap
        bx = x - total / 2 + bw / 2

        for lib in libs:
            bc = BADGE_BG.get(lib, "#30363d") if unlocked(node, week) else "#30363d"
            ax.add_patch(
                FancyBboxPatch(
                    (bx - bw / 2, badge_cy - 0.20),
                    bw,
                    0.38,
                    boxstyle="round,pad=0.06",
                    lw=0,
                    fc=bc,
                    alpha=0.92 * max(alpha, 0.24),
                    zorder=5,
                )
            )
            ax.text(
                bx,
                badge_cy,
                lib,
                ha="center",
                va="center",
                fontsize=5.0,
                fontweight="bold",
                color=TEXT_BRIGHT if unlocked(node, week) else LOCKED_TEXT,
                alpha=max(alpha, 0.45),
                zorder=6,
            )
            bx += bw + gap

    draw_week_badge(ax, node, week, future, x + MW - 0.20, y - MH + 0.78)


def render_tree(
    tree: dict,
    out_base: Path,
    week: int | None,
    future: str,
    highlight_current: bool,
    formats: list[str],
    dpi: int,
):
    cw = tree.get("canvas", {}).get("width", 76)
    ch = tree.get("canvas", {}).get("height", 52)

    fig, ax = plt.subplots(figsize=(28, 18))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, cw)
    ax.set_ylim(0, ch)
    ax.set_aspect("equal")
    ax.axis("off")

    xs = np.arange(1.5, cw, 1.5)
    ys = np.arange(1.5, ch, 1.5)
    gx, gy = np.meshgrid(xs, ys)
    ax.scatter(gx.ravel(), gy.ravel(), s=0.6, color="#1c2128", zorder=0)

    title = tree.get("title", "DATA MINING & MACHINE LEARNING")
    subtitle = tree.get("subtitle", "Model Selection Skill Tree")
    if week is not None:
        subtitle = f"{subtitle} - Week {week:02d}"
        if highlight_current:
            subtitle += " highlighted"

    ax.text(
        cw / 2,
        51.4,
        title,
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=TEXT_BRIGHT,
        fontfamily="monospace",
        zorder=5,
        path_effects=[pe.withStroke(linewidth=3, foreground=BG)],
    )
    ax.text(cw / 2, 50.2, subtitle, ha="center", va="top", fontsize=9, color=TEXT_DIM, zorder=5)
    ax.plot([1, cw - 1], [49.5, 49.5], color="#21262d", lw=1.0, zorder=1)

    for src, dst in tree["edges"]:
        bezier_edge(ax, tree, src, dst, week, future)

    for node in tree["nodes"]:
        if node.get("kind") == "header":
            header_node(ax, node, week, future, highlight_current)
        else:
            method_node(ax, node, week, future, highlight_current)

    shap_col = branch_col("shap_special")
    ax.add_patch(
        FancyBboxPatch(
            (1.2, 1.5),
            0.7,
            0.7,
            boxstyle="round,pad=0.06",
            lw=1.2,
            ec=shap_col,
            fc="#1a1608",
            alpha=0.9,
            zorder=5,
            linestyle=(0, (5, 3)),
        )
    )
    ax.text(
        2.2,
        1.85,
        "Dashed gold border = cross-cutting tool (applies to any model)",
        fontsize=6.0,
        color=TEXT_DIM,
        va="center",
        zorder=5,
    )

    plt.tight_layout(pad=0.3)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        target = out_base.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight", "facecolor": BG}
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(target, format=fmt, **save_kwargs)
        written.append(target)
    plt.close(fig)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the DMML skill tree.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to tree JSON data.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Directory for rendered files.")
    parser.add_argument("--week", type=int, help="Render the tree as of a specific week.")
    parser.add_argument("--full", action="store_true", help="Render the complete tree.")
    parser.add_argument("--all-weeks", action="store_true", help="Render one image per week.")
    parser.add_argument(
        "--future",
        choices=["dim", "hide"],
        default="dim",
        help="For weekly renders, dim or hide future nodes.",
    )
    parser.add_argument(
        "--highlight-current",
        action="store_true",
        help="Add an extra glow around nodes introduced in the selected week.",
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=130, help="PNG output DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tree = load_tree(args.data)
    max_week = max(int(node.get("week", 0)) for node in tree["nodes"])

    jobs: list[tuple[int | None, str]] = []
    if args.all_weeks:
        jobs.extend((week, f"week_{week:02d}") for week in range(1, max_week + 1))
    if args.week is not None:
        jobs.append((args.week, f"week_{args.week:02d}"))
    if args.full or not jobs:
        jobs.append((None, "full"))

    for week, stem in jobs:
        written = render_tree(
            tree=tree,
            out_base=args.out_dir / stem,
            week=week,
            future=args.future,
            highlight_current=args.highlight_current,
            formats=args.formats,
            dpi=args.dpi,
        )
        for path in written:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
