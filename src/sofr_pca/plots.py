"""Charts: static PNGs (matplotlib) for the README and interactive versions (plotly) for reports/report.html.

One style everywhere: white surface, hairline grid, a fixed categorical colour order, a blue <-> red
diverging scale with a grey midpoint, titles that state the finding, units on every axis and a source note.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

if "ipykernel" not in sys.modules:      # scripts render off-screen; notebooks keep their inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots

# ---------------------------------------------------------------- palette (fixed order, never cycled)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, RED, MID = "#2a78d6", "#e34948", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("bgr", [BLUE, MID, RED])
PL_DIVERGING = [[0, BLUE], [0.5, MID], [1, RED]]
SOURCE = "Source: Bloomberg settlement data; author's calculations"
REGIME_SHADE = {"hiking": ("#eb6834", 0.08), "cutting": ("#2a78d6", 0.08), "2026": ("#6250d6", 0.07)}
DPI = 160


def apply_style() -> None:
    """Global matplotlib style: quiet axes, readable fonts, white surface."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": SURFACE, "savefig.facecolor": "white",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "axes.axisbelow": True, "legend.fontsize": 8.5, "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"], "lines.linewidth": 1.8,
    })


def _finish(fig, title: str, path: pathlib.Path | None) -> plt.Figure:
    """Finding-style title on top, source note bottom-left, save at >= 150 dpi."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, SOURCE, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    if path is not None:
        fig.savefig(path, dpi=DPI)
    return fig


def _events(ax, events: dict, lo, hi, y: float = 0.98) -> None:
    for name, d in events.items():
        d = pd.Timestamp(d)
        if lo <= d <= hi:
            ax.axvline(d, color=MUTED, lw=0.8, ls=":")
            ax.text(d, y, " " + name, transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, va="top", rotation=90)


def _regimes(ax, regimes: list, lo, hi) -> None:
    done = set()
    for label, a, b in regimes:
        if label in REGIME_SHADE:
            a, b = max(pd.Timestamp(a), lo), min(pd.Timestamp(b), hi)
            if a < b:
                col, al = REGIME_SHADE[label]
                ax.axvspan(a, b, color=col, alpha=al, lw=0, label=f"{label} regime" if label not in done else None)
                done.add(label)


# ---------------------------------------------------------------- 1. explained variance + loadings

def fig_pca(full, path=None):
    ex = full.explained[:5]
    L = full.loadings
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 2]})
    a1.bar([f"PC{i + 1}" for i in range(5)], ex * 100, color=C[0], width=0.6)
    for i, v in enumerate(ex):
        a1.text(i, v * 100 + 1.5, f"{v:.1%}", ha="center", fontsize=8.5, color=INK2)
    a1.set_ylabel("share of daily variance (%)"); a1.set_ylim(0, 105); a1.set_title("Explained variance", loc="left")
    names = {"PC1": "PC1 level", "PC2": "PC2 slope", "PC3": "PC3 curvature"}
    for j, c in enumerate(L.columns):
        a2.plot(L.index, L[c], color=C[j], marker="o", ms=4, label=names.get(c, c))
        a2.text(L.index[-1] + 0.15, L[c].iloc[-1], names.get(c, c), color=INK2, fontsize=8.5, va="center")
    a2.axhline(0, color=AXIS, lw=0.8)
    a2.set_xticks(L.index); a2.set_xlabel("curve position (1 = front quarterly contract, 12 = three years out)")
    a2.set_ylabel("loading (unit-length vector)"); a2.set_xlim(0.7, 13.6)
    a2.set_title("What each component looks like across the curve", loc="left"); a2.legend(loc="lower right")
    t = f"One factor (level) explains {ex[0]:.0%} of daily SOFR curve moves; slope and curvature add {ex[1] + ex[2]:.0%}"
    return _finish(fig, t, path)


def plotly_pca(full) -> go.Figure:
    ex, L = full.explained[:5], full.loadings
    f = make_subplots(1, 2, column_widths=[0.33, 0.67], subplot_titles=("Explained variance (%)", "Loadings by curve position"))
    f.add_bar(x=[f"PC{i + 1}" for i in range(5)], y=ex * 100, marker_color=C[0], name="explained %",
              hovertemplate="%{x}: %{y:.1f}%<extra></extra>", row=1, col=1)
    for j, c in enumerate(L.columns):
        f.add_scatter(x=L.index, y=L[c], mode="lines+markers", name=c, line=dict(color=C[j], width=2), row=1, col=2,
                      hovertemplate="position %{x}: %{y:.3f}<extra>" + c + "</extra>")
    f.update_xaxes(title_text="curve position", row=1, col=2)
    return _plotly_layout(f, f"One factor explains {ex[0]:.0%} of daily curve moves; slope and curvature add {ex[1] + ex[2]:.0%}")


# ---------------------------------------------------------------- 2. stability

def fig_regime_loadings(by_regime: dict, path=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    names = ["PC1 level", "PC2 slope", "PC3 curvature"]
    for k, ax in enumerate(axes):
        for j, (label, m) in enumerate(by_regime.items()):
            ax.plot(m.loadings.index, m.loadings.iloc[:, k], color=C[j], lw=1.6, marker="o", ms=3,
                    label=f"{label} ({m.explained[k]:.0%})")
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(names[k], loc="left"); ax.set_xticks(range(1, 13)); ax.set_xlabel("curve position")
        if k == 0:                                   # one legend: colours are the same periods in all panels
            ax.legend(fontsize=7.5, loc="lower right", title="period (PC1 share of variance)", title_fontsize=7.5)
    axes[0].set_ylabel("loading")
    t = "Factor shapes have been stable since 2022; the zero-bound years (2020-21) are the outlier, when the front stopped moving"
    return _finish(fig, t, path)


def fig_rolling_explained(roll: pd.DataFrame, events: dict, path=None):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for j, c in enumerate(roll.columns):
        ax.plot(roll.index, roll[c] * 100, color=C[j], label=c)
    _events(ax, events, roll.index[0], roll.index[-1])
    ax.set_ylabel("share of daily variance (%)"); ax.legend(loc="center left")
    t = f"PC1's share of daily moves ranges {roll['PC1'].min():.0%}-{roll['PC1'].max():.0%} on rolling 1-year windows"
    return _finish(fig, t, path)


def plotly_regime_loadings(by_regime: dict, roll: pd.DataFrame) -> go.Figure:
    f = make_subplots(2, 3, specs=[[{}, {}, {}], [{"colspan": 3}, None, None]], row_heights=[0.55, 0.45],
                      subplot_titles=("PC1 level by period", "PC2 slope by period", "PC3 curvature by period",
                                      "Explained variance, rolling 252-day window (%)"))
    for k in range(3):
        for j, (label, m) in enumerate(by_regime.items()):
            f.add_scatter(x=m.loadings.index, y=m.loadings.iloc[:, k], mode="lines+markers", name=label, legendgroup=label,
                          showlegend=k == 0, line=dict(color=C[j], width=2), row=1, col=k + 1,
                          hovertemplate="position %{x}: %{y:.3f}<extra>" + label + "</extra>")
    for j, c in enumerate(roll.columns):
        f.add_scatter(x=roll.index, y=roll[c] * 100, mode="lines", name=f"{c} share", line=dict(color=C[j + 5], width=2),
                      row=2, col=1, hovertemplate="%{x|%d-%b-%Y}: %{y:.1f}%<extra>" + c + "</extra>")
    return _plotly_layout(f, "Level, slope and curvature are stable across regimes except at the zero bound", height=720)


# ---------------------------------------------------------------- 3. z heatmap

def fig_heatmap(z: pd.DataFrame, names: list[str], path=None):
    zz = z[names].T
    fig, ax = plt.subplots(figsize=(12, 5.2))
    im = ax.imshow(zz.clip(-3, 3).to_numpy(), aspect="auto", cmap=DIVERGING, vmin=-3, vmax=3, interpolation="nearest",
                   extent=(matplotlib.dates.date2num(zz.columns[0]), matplotlib.dates.date2num(zz.columns[-1]), len(names) - 0.5, -0.5))
    ax.xaxis_date(); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.5); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("z-score (red = rate above fair value = cheap)", fontsize=8.5)
    share = float((zz.abs() > 2).to_numpy().mean())
    t = f"Out-of-sample fly z-scores: dislocations beyond |z| = 2 occur on {share:.0%} of fly-days, are brief, and scatter along the curve"
    return _finish(fig, t, path)


def plotly_heatmap(z: pd.DataFrame, names: list[str]) -> go.Figure:
    zz = z[names].T
    f = go.Figure(go.Heatmap(z=zz.clip(-3, 3).to_numpy(), x=zz.columns, y=names, colorscale=PL_DIVERGING, zmin=-3, zmax=3,
                             colorbar=dict(title="z"), hovertemplate="%{y}<br>%{x|%d-%b-%Y}<br>z = %{z:.2f}<extra></extra>"))
    f.update_yaxes(autorange="reversed")
    return _plotly_layout(f, "Out-of-sample z-scores of every fly and double fly", height=560)


# ---------------------------------------------------------------- 4. worked example

def fig_example(level: pd.Series, anchor: pd.Series, z: pd.Series, trades: pd.DataFrame, name: str, p, path=None):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    a1.plot(level.index, level, color=C[0], lw=1.3, label=f"{name} level")
    a1.plot(anchor.index, anchor, color=C[1], lw=1.3, label="model fair value (frozen PCA fit + trailing residual mean)")
    a1.set_ylabel("fly level (bp, rate terms)"); a1.legend(loc="upper left")
    a2.plot(z.index, z, color=C[0], lw=1.1)
    for v, ls in [(p.entry_z, "--"), (-p.entry_z, "--"), (p.exit_z, ":"), (-p.exit_z, ":")]:
        a2.axhline(v, color=MUTED, lw=0.8, ls=ls)
    a2.axhline(0, color=AXIS, lw=0.8)
    if len(trades):
        for _, r in trades.iterrows():
            e, x = pd.Timestamp(r.entry_date), pd.Timestamp(r.exit_date)
            if e in z.index:
                a2.scatter(e, z.loc[e], color=C[2] if r.gross_bp > 0 else C[7], s=30, zorder=3, edgecolor=SURFACE, lw=1.5)
            if x in z.index:
                a2.scatter(x, z.loc[x], color=MUTED, marker="x", s=22, zorder=3)
    a2.set_ylabel("residual z-score"); a2.set_ylim(-5, 5)
    a2.text(0.005, 0.03, "dots = entries (green made money before costs, red lost); x = exits; dashed = entry band, dotted = exit band",
            transform=a2.transAxes, fontsize=7.5, color=INK2)
    n, wins = len(trades), int((trades.gross_bp > 0).sum()) if len(trades) else 0
    gross = trades.gross_bp.mean() if n else 0.0
    t = f"{name}: {n} trades in the window, {wins} winners before costs, average {gross:+.1f} bp gross per trade"
    return _finish(fig, t, path)


def plotly_example(level, anchor, z, trades, name, p) -> go.Figure:
    f = make_subplots(2, 1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06,
                      subplot_titles=(f"{name} level vs model fair value (bp)", "Residual z-score"))
    f.add_scatter(x=level.index, y=level, name="level", line=dict(color=C[0], width=1.5), row=1, col=1)
    f.add_scatter(x=anchor.index, y=anchor, name="fair value", line=dict(color=C[1], width=1.5), row=1, col=1)
    f.add_scatter(x=z.index, y=z, name="z", line=dict(color=C[0], width=1.2), row=2, col=1, showlegend=False)
    if len(trades):
        e = trades[pd.to_datetime(trades.entry_date).isin(z.index)]
        f.add_scatter(x=pd.to_datetime(e.entry_date), y=z.reindex(pd.to_datetime(e.entry_date)), mode="markers", name="entries",
                      marker=dict(size=9, color=[C[2] if g > 0 else C[7] for g in e.gross_bp], line=dict(color="white", width=1.5)),
                      text=[f"gross {g:+.2f} bp, exit: {r}" for g, r in zip(e.gross_bp, e.exit_reason)],
                      hovertemplate="%{x|%d-%b-%Y} z=%{y:.2f}<br>%{text}<extra></extra>", row=2, col=1)
    for v in (p.entry_z, -p.entry_z):
        f.add_hline(y=v, line=dict(color=MUTED, dash="dash", width=1), row=2, col=1)
    return _plotly_layout(f, f"Worked example: {name}", height=620)


# ---------------------------------------------------------------- 5. half-life by position

def fig_half_life(stats: pd.DataFrame, path=None):
    fams = [("fly", "flies"), ("dfly", "double flies"), ("spread3m", "3M spreads")]
    fig, ax = plt.subplots(figsize=(11, 4))
    for j, (f, lab) in enumerate(fams):
        s = stats[stats.family == f]
        ax.plot(s.centre, s.half_life_wf_median, color=C[j], marker="o", ms=6, label=lab)
        ax.text(s.centre.iloc[-1] + 0.15, s.half_life_wf_median.iloc[-1], lab, color=INK2, fontsize=8.5, va="center")
    ax.set_xlabel("centre of the structure on the curve (position)"); ax.set_ylabel("median half-life (business days)")
    ax.set_xticks(range(1, 13)); ax.set_xlim(1, 12.9); ax.set_ylim(0, None); ax.legend(loc="upper center")
    fl = stats[stats.family == "fly"]
    front = fl[fl.centre <= 5].half_life_wf_median.median(); back = fl[fl.centre >= 9].half_life_wf_median.median()
    t = f"Residuals halve fast: flies in ~{front:.0f} days at the front and belly, ~{back:.0f} days at the back"
    return _finish(fig, t, path)


def plotly_half_life(stats: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    for j, (fam, lab) in enumerate([("fly", "flies"), ("dfly", "double flies"), ("spread3m", "3M spreads"), ("spread1y", "1Y spreads")]):
        s = stats[stats.family == fam]
        f.add_bar(x=s.centre, y=s.half_life_wf_median, name=lab, marker_color=C[j], text=s.index,
                  hovertemplate="%{text}: %{y:.1f} days<extra></extra>")
    f.update_layout(barmode="group", xaxis_title="centre position", yaxis_title="median half-life (days)")
    return _plotly_layout(f, "Median half-life of residuals across walk-forward windows")


# ---------------------------------------------------------------- 6. equity curve

def fig_equity(res, regimes: list, events: dict, path=None, alt: pd.Series | None = None, alt_label: str = ""):
    net, gross = res.daily_pnl.cumsum(), res.daily_gross.cumsum()
    dd = net - net.cummax()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    _regimes(a1, regimes, net.index[0], net.index[-1])
    a1.plot(gross.index, gross / 1e3, color=C[0], label="gross (before costs)")
    if alt is not None:
        a1.plot(alt.index, alt.cumsum() / 1e3, color=C[2], label=alt_label)
    a1.plot(net.index, net / 1e3, color=C[1], label="net of costs (full tick per leg + $2.50/contract)")
    a1.axhline(0, color=AXIS, lw=0.8); a1.set_ylabel("cumulative P&L ($k, 1 unit per trade)")
    _events(a1, events, net.index[0], net.index[-1]); a1.legend(loc="lower left")
    a2.fill_between(dd.index, dd / 1e3, 0, color=C[7], alpha=0.25, lw=0); a2.plot(dd.index, dd / 1e3, color=C[7], lw=1)
    a2.set_ylabel("net drawdown ($k)")
    from .backtest import sharpe
    t = (f"Walk-forward, out-of-sample: before costs the fly book roughly breaks even (Sharpe {sharpe(res.daily_gross):.1f}); "
         f"after costs it loses ${-res.daily_pnl.sum() / 1e3:.0f}k")
    return _finish(fig, t, path)


def plotly_equity(res, regimes: list) -> go.Figure:
    net, gross = res.daily_pnl.cumsum(), res.daily_gross.cumsum()
    f = make_subplots(2, 1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05,
                      subplot_titles=("Cumulative P&L ($)", "Net drawdown ($)"))
    f.add_scatter(x=gross.index, y=gross, name="gross", line=dict(color=C[0], width=2), row=1, col=1)
    f.add_scatter(x=net.index, y=net, name="net of costs", line=dict(color=C[1], width=2), row=1, col=1)
    f.add_scatter(x=net.index, y=net - net.cummax(), name="drawdown", fill="tozeroy", line=dict(color=C[7], width=1),
                  row=2, col=1)
    for label, a, b in regimes:
        if label in REGIME_SHADE and pd.Timestamp(b) >= net.index[0]:
            f.add_vrect(x0=max(pd.Timestamp(a), net.index[0]), x1=pd.Timestamp(b), fillcolor=REGIME_SHADE[label][0],
                        opacity=0.08, line_width=0, row=1, col=1)
    return _plotly_layout(f, "Walk-forward equity curve (orange shading = hiking, blue = cutting, violet = 2026)", height=600)


# ---------------------------------------------------------------- 7. robustness

def fig_robustness(grid: pd.DataFrame, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, lab in [(axes[0], "sharpe_net", "Net Sharpe (after costs)"), (axes[1], "sharpe_gross", "Gross Sharpe (before costs)")]:
        m = grid.pivot(index="window", columns="entry_z", values=col)
        lim = max(1.0, float(np.nanmax(np.abs(m.to_numpy()))))
        ax.imshow(m.to_numpy(), cmap=DIVERGING.reversed(), vmin=-lim, vmax=lim, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.iat[i, j]:.2f}", ha="center", va="center", fontsize=9, color=INK)
        ax.set_xticks(range(m.shape[1])); ax.set_xticklabels([f"{v:g}" for v in m.columns])
        ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index)
        ax.set_xlabel("entry threshold |z|"); ax.set_ylabel("estimation window (days)"); ax.set_title(lab, loc="left"); ax.grid(False)
    pos = int((grid.sharpe_net > 0).sum())
    t = (f"No setting survives costs: net Sharpe is positive in {pos} of {len(grid)} combinations; "
         f"the best gross Sharpe is {grid.sharpe_gross.max():.2f}")
    return _finish(fig, t, path)


def plotly_robustness(grid: pd.DataFrame) -> go.Figure:
    f = make_subplots(1, 2, subplot_titles=("Net Sharpe", "Gross Sharpe"))
    for k, col in enumerate(["sharpe_net", "sharpe_gross"]):
        m = grid.pivot(index="window", columns="entry_z", values=col)
        lim = max(1.0, float(np.nanmax(np.abs(m.to_numpy()))))
        f.add_heatmap(z=m.to_numpy(), x=[f"|z|>{v:g}" for v in m.columns], y=[f"{w}d" for w in m.index],
                      colorscale=[[0, RED], [0.5, MID], [1, BLUE]], zmin=-lim, zmax=lim, showscale=False,
                      text=np.round(m.to_numpy(), 2), texttemplate="%{text}", row=1, col=k + 1,
                      hovertemplate="%{y}, %{x}: %{z:.2f}<extra></extra>")
    return _plotly_layout(f, "Robustness: entry threshold x estimation window", height=380)


# ---------------------------------------------------------------- 8. today's ranking

def fig_today(top: pd.DataFrame, entry_z: float, path=None):
    d = top.sort_values("z")
    fig, ax = plt.subplots(figsize=(10, 6))
    cols = [BLUE if z < 0 else RED for z in d.z]
    ax.barh(d.structure + "  (" + d.contracts + ")", d.z, color=cols, height=0.65)
    for v in (entry_z, -entry_z):
        ax.axvline(v, color=MUTED, ls="--", lw=0.9)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("residual z-score (left/blue = rich: sell the package; right/red = cheap: buy the package)")
    ax.tick_params(axis="y", labelsize=7.5)
    n = int((top.z.abs() > entry_z).sum())
    date = pd.Timestamp(top.date.iloc[0]).strftime("%d-%b-%Y")
    t = (f"{date}: {n} structure{'s' if n != 1 else ''} past the |z| > {entry_z:g} entry line - "
         + ("the curve is close to fair value" if n == 0 else "candidates shown in the table"))
    return _finish(fig, t, path)


def plotly_today(top: pd.DataFrame) -> go.Figure:
    d = top.sort_values("z")
    f = go.Figure(go.Bar(x=d.z, y=d.structure, orientation="h", marker_color=[BLUE if z < 0 else RED for z in d.z],
                         text=d.contracts, hovertemplate="%{y} (%{text}): z = %{x:.2f}<extra></extra>"))
    f.update_layout(xaxis_title="z-score (blue = rich, red = cheap)")
    return _plotly_layout(f, "Today's richest and cheapest structures", height=560)


# ---------------------------------------------------------------- 9. signal vs cost (hero)

def fig_signal_vs_cost(sig: pd.DataFrame, horizon: int, path=None):
    s = sig[sig.horizon_days == horizon]
    fams = [("fly", "flies"), ("dfly", "double flies"), ("spread3m", "3M spreads"), ("spread1y", "1Y spreads")]
    buckets = list(dict.fromkeys(s.z_bucket))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.2
    for j, (f, lab) in enumerate(fams):
        r = s[s.family == f].set_index("z_bucket").reindex(buckets)
        ax.bar(np.arange(len(buckets)) + (j - 1.5) * w, r.avg_reversion_bp, width=w - 0.02, color=C[j], label=lab)
    for f, lab, col in [("fly", "round-trip cost of one fly (4 legs)", C[0]), ("dfly", "round-trip cost of one double fly (8 legs)", C[1])]:
        c = s[s.family == f].round_trip_cost_bp.iloc[0]
        ax.axhline(c, color=col, ls="--", lw=1.2); ax.text(len(buckets) - 0.5, c, f"{lab}: {c:.1f} bp", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(range(len(buckets))); ax.set_xticklabels([f"|z| {b}" for b in buckets])
    ax.set_ylabel(f"average reversion over the next {horizon} days (bp)"); ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    cur = s[s.family.isin(["fly", "dfly"])]
    t = (f"Curvature does mean-revert ({cur.avg_reversion_bp.min():.1f}-{cur.avg_reversion_bp.max():.1f} bp in {horizon} days, "
         f"hit rate {cur.hit_rate.min():.0%}-{cur.hit_rate.max():.0%}) - but crossing every leg costs "
         f"{cur.round_trip_cost_bp.min():.1f}-{cur.round_trip_cost_bp.max():.1f} bp")
    return _finish(fig, t, path)


def plotly_signal(sig: pd.DataFrame) -> go.Figure:
    f = make_subplots(1, len(sig.horizon_days.unique()), shared_yaxes=True,
                      subplot_titles=[f"next {h} days" for h in sorted(sig.horizon_days.unique())])
    for k, h in enumerate(sorted(sig.horizon_days.unique())):
        for j, fam in enumerate(["fly", "dfly", "spread3m", "spread1y"]):
            r = sig[(sig.horizon_days == h) & (sig.family == fam)]
            f.add_bar(x=r.z_bucket, y=r.avg_reversion_bp, name=fam, marker_color=C[j], legendgroup=fam, showlegend=k == 0,
                      customdata=np.c_[r.hit_rate, r.signals], row=1, col=k + 1,
                      hovertemplate=fam + " |z| %{x}: %{y:.2f} bp<br>hit %{customdata[0]:.0%}, n=%{customdata[1]}<extra></extra>")
    f.update_layout(barmode="group", yaxis_title="average reversion (bp)")
    return _plotly_layout(f, "Signal quality: average out-of-sample reversion after a signal (costs: fly 4.8 bp, double fly 9.6 bp round trip)", height=420)


# ---------------------------------------------------------------- report helpers

def _plotly_layout(f: go.Figure, title: str, height: int = 460) -> go.Figure:
    f.update_layout(title=dict(text=title, x=0.01, font=dict(size=15, color=INK)), template="plotly_white", height=height,
                    font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=12, color=INK2),
                    margin=dict(l=60, r=30, t=70, b=50), legend=dict(orientation="h", y=-0.12),
                    plot_bgcolor=SURFACE, colorway=C, hoverlabel=dict(bgcolor="white"))
    f.update_xaxes(gridcolor=GRID, linecolor=AXIS); f.update_yaxes(gridcolor=GRID, linecolor=AXIS)
    return f


def table_html(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    """Plain HTML table with right-aligned numbers."""
    return df.to_html(classes="tbl", border=0, float_format=lambda x: floatfmt.format(x), na_rep="-")


def write_report(path: pathlib.Path, sections: list[tuple[str, str, object]], title: str, intro: str) -> None:
    """One self-contained HTML page: each section = (heading, text, figure-or-html)."""
    parts, first = [], True
    for heading, text, obj in sections:
        if isinstance(obj, go.Figure):
            body = obj.to_html(full_html=False, include_plotlyjs="cdn" if first else False, config={"displaylogo": False})
            first = False
        else:
            body = str(obj)
        parts.append(f"<section><h2>{heading}</h2><p>{text}</p>{body}</section>")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--ink:{INK};--ink2:{INK2};--muted:{MUTED};--grid:{GRID};--bg:#f9f9f7;--surface:#ffffff}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Segoe UI",Helvetica,Arial,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px 16px 60px}}
h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:18px;margin:0 0 6px}}
.intro{{color:var(--ink2);max-width:900px}} p{{color:var(--ink2);max-width:960px}}
section{{background:var(--surface);border:1px solid rgba(11,11,11,.08);border-radius:10px;padding:18px 18px 10px;margin:18px 0;overflow-x:auto}}
.tbl{{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;margin:6px 0 10px}}
.tbl th,.tbl td{{padding:5px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}} .tbl thead th{{color:var(--ink2);font-weight:600}}
footer{{color:var(--muted);font-size:12px;margin-top:30px}}
</style></head><body><main><h1>{title}</h1><p class="intro">{intro}</p>{''.join(parts)}
<footer>{SOURCE}. Research project; not investment advice.</footer></main></body></html>"""
    path.write_text(html, encoding="utf-8")
