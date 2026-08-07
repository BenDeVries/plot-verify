#!/usr/bin/env python3
"""Generate a test set of Agent JSONs (schema 1.1), one per plot type.

For each plot type:
1. synthesize ground-truth data,
2. render it with matplotlib -> the PNG becomes the embedded source image,
3. read exact pixel/value calibration pairs off the matplotlib transform,
4. jitter the values slightly to mimic digitization error,
5. write the JSON (image data_uri + axes + rows + series).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/benjamin/plot-verify/test_jsons")
PNG = OUT / "png"
OUT.mkdir(exist_ok=True)
PNG.mkdir(exist_ok=True)

DPI = 100
rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_fig():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    return fig, ax


def finish(fig, ax, name):
    """Draw, compute the pixel transform, save the PNG.

    Returns (px_of, width, height, png_bytes) where px_of(x, y) maps data
    coords to image pixel coords (origin top-left, y down).
    """
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    def px_of(x, y):
        dx, dy = ax.transData.transform((float(x), float(y)))
        return float(dx), float(h - dy)

    path = PNG / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return px_of, w, h, path.read_bytes()


def axes_block(px_of, x_pair, y_pair, *, x_scale="linear", y_scale="linear",
               x_log_base=None, y_log_base=None):
    """Two exact pixel/value pairs per axis, read off the transform."""
    x_lo, x_hi = x_pair
    y_lo, y_hi = y_pair
    ax_x = {
        "scale": x_scale,
        "calibration": [
            {"pixel": round(px_of(x_lo, y_lo)[0], 2), "value": x_lo},
            {"pixel": round(px_of(x_hi, y_lo)[0], 2), "value": x_hi},
        ],
    }
    ax_y = {
        "scale": y_scale,
        "calibration": [
            {"pixel": round(px_of(x_lo, y_lo)[1], 2), "value": y_lo},
            {"pixel": round(px_of(x_lo, y_hi)[1], 2), "value": y_hi},
        ],
    }
    if x_log_base is not None:
        ax_x["log_base"] = x_log_base
    if y_log_base is not None:
        ax_y["log_base"] = y_log_base
    return {"x": ax_x, "y": ax_y}


def jit(values, frac, span):
    """Jitter: gaussian noise scaled to a fraction of the axis span."""
    arr = np.asarray(values, dtype=float)
    return arr + rng.normal(0.0, frac * span, size=arr.shape)


def write_json(name, *, plot_type, orientation, image_bytes, w, h,
               axes, rows, series):
    doc = {
        "schema_version": "1.1",
        "image": {
            "filename": f"{name}.png",
            "width_px": w,
            "height_px": h,
            "data_uri": "data:image/png;base64,"
                        + base64.b64encode(image_bytes).decode("ascii"),
        },
        "plot_type": plot_type,
        "orientation": orientation,
        "axes": axes,
        "rows": rows,
        "series": series,
    }
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(doc, indent=2))
    print(f"wrote {path}  ({len(rows)} rows, image {w}x{h})")


def r6(v):
    return round(float(v), 6)


def ordered_bounds(center, lo, hi):
    """Keep jittered bounds bracketing the jittered center."""
    lo2 = min(lo, center - 1e-9)
    hi2 = max(hi, center + 1e-9)
    return lo2, hi2


# ---------------------------------------------------------------------------
# 1. Scatter — two correlated series
# ---------------------------------------------------------------------------

def make_scatter():
    n = 18
    x_a = rng.uniform(1, 9, n)
    y_a = 2.0 + 0.9 * x_a + rng.normal(0, 0.8, n)
    x_b = rng.uniform(1, 9, n)
    y_b = 9.5 - 0.55 * x_b + rng.normal(0, 0.9, n)

    fig, ax = new_fig()
    ax.scatter(x_a, y_a, color="#d62728", label="Drug A", s=45)
    ax.scatter(x_b, y_b, color="#1f77b4", label="Drug B", s=45, marker="s")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_xlabel("Dose (mg)")
    ax.set_ylabel("Response")
    ax.legend()
    ax.grid(alpha=0.3)
    px_of, w, h, png = finish(fig, ax, "scatter")

    rows = []
    for xs, ys, s, c in ((x_a, y_a, "Drug A", "#d62728"),
                         (x_b, y_b, "Drug B", "#1f77b4")):
        xj = jit(xs, 0.004, 10.0)
        yj = jit(ys, 0.004, 14.0)
        rows += [{"series": s, "x": r6(a), "y": r6(b), "series_color": c}
                 for a, b in zip(xj, yj)]

    write_json("scatter", plot_type="scatter", orientation="vertical",
               image_bytes=png, w=w, h=h,
               axes=axes_block(px_of, (0, 10), (0, 14)),
               rows=rows,
               series=[{"key": "Drug A", "color": "#d62728"},
                       {"key": "Drug B", "color": "#1f77b4"}])


# ---------------------------------------------------------------------------
# 2. Time series with intervals — two series, SD band
# ---------------------------------------------------------------------------

def make_time_series():
    t = np.arange(0, 25, 3)
    y_a = 40 + 25 * np.exp(-t / 10.0) + rng.normal(0, 1.0, t.size)
    sd_a = 4.0 + rng.uniform(-0.5, 0.5, t.size)
    y_b = 35 + 0.9 * t + rng.normal(0, 1.0, t.size)
    sd_b = 5.0 + rng.uniform(-0.5, 0.5, t.size)

    fig, ax = new_fig()
    for y, sd, c, lbl in ((y_a, sd_a, "#2ca02c", "Control"),
                          (y_b, sd_b, "#9467bd", "Treatment")):
        ax.errorbar(t, y, yerr=sd, color=c, label=lbl, marker="o",
                    capsize=4, lw=1.8)
        ax.fill_between(t, y - sd, y + sd, color=c, alpha=0.15)
    ax.set_xlim(-1, 26)
    ax.set_ylim(20, 75)
    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)
    px_of, w, h, png = finish(fig, ax, "time_series")

    rows = []
    for y, sd, s, c in ((y_a, sd_a, "Control", "#2ca02c"),
                        (y_b, sd_b, "Treatment", "#9467bd")):
        for i, tv in enumerate(t):
            yj = float(jit([y[i]], 0.004, 55.0)[0])
            lo = float(jit([y[i] - sd[i]], 0.004, 55.0)[0])
            hi = float(jit([y[i] + sd[i]], 0.004, 55.0)[0])
            lo, hi = ordered_bounds(yj, lo, hi)
            rows.append({"series": s, "x": r6(tv), "y": r6(yj),
                         "y_err_lower": r6(lo), "y_err_upper": r6(hi),
                         "error_bar_type": "SD", "series_color": c})

    write_json("time_series", plot_type="line_timeseries",
               orientation="vertical", image_bytes=png, w=w, h=h,
               axes=axes_block(px_of, (0, 24), (20, 70)),
               rows=rows,
               series=[{"key": "Control", "color": "#2ca02c"},
                       {"key": "Treatment", "color": "#9467bd"}])


# ---------------------------------------------------------------------------
# 3. Forest — odds ratios on a log x-axis, summary diamond
# ---------------------------------------------------------------------------

def make_forest():
    labels = ["Study A", "Study B", "Study C", "Study D", "Study E",
              "Pooled"]
    or_vals = np.array([0.62, 1.35, 0.80, 2.10, 0.95, 1.02])
    ci_lo = or_vals * np.array([0.38, 0.72, 0.55, 1.15, 0.60, 0.85])
    ci_hi = or_vals * np.array([1.65, 1.90, 1.45, 1.80, 1.55, 1.22])
    is_summary = [False] * 5 + [True]
    n = len(labels)

    fig, ax = new_fig()
    for i in range(n):
        y = n - 1 - i
        if is_summary[i]:
            ax.plot(or_vals[i], y, marker="D", color="#111", ms=11)
        else:
            ax.plot([ci_lo[i], ci_hi[i]], [y, y], color="#333", lw=1.6)
            ax.plot(or_vals[i], y, marker="s", color="#333", ms=7)
    ax.axvline(1.0, color="#999", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 5.0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_xlabel("Odds ratio (log scale)")
    px_of, w, h, png = finish(fig, ax, "forest")

    rows = []
    for i in range(n):
        # multiplicative jitter on a log axis
        f = float(np.exp(rng.normal(0, 0.012)))
        v = or_vals[i] * f
        lo = ci_lo[i] * float(np.exp(rng.normal(0, 0.012)))
        hi = ci_hi[i] * float(np.exp(rng.normal(0, 0.012)))
        lo, hi = ordered_bounds(v, lo, hi)
        rows.append({"series": labels[i], "value": r6(v),
                     "value_err_lower": r6(lo), "value_err_upper": r6(hi),
                     "is_summary": bool(is_summary[i]),
                     "status": "pooled estimate" if is_summary[i] else "",
                     "error_bar_type": "Confidence"})

    write_json("forest", plot_type="forest", orientation="horizontal",
               image_bytes=png, w=w, h=h,
               axes=axes_block(px_of, (0.25, 4.0), (0, n - 1),
                               x_scale="log", x_log_base=10),
               rows=rows,
               series=[{"key": lb} for lb in labels])


# ---------------------------------------------------------------------------
# 4/5. Bar — vertical and horizontal variants
# ---------------------------------------------------------------------------

_BAR_CATS = [1, 2, 3, 4, 5]
_BAR_VALS = np.array([12.0, 19.5, 8.2, 15.1, 22.3])
_BAR_SD = np.array([1.8, 2.4, 1.2, 2.0, 2.6])


def _bar_rows(vals, sd, horizontal):
    rows = []
    for i, cat in enumerate(_BAR_CATS):
        vj = float(jit([vals[i]], 0.005, 25.0)[0])
        lo = float(jit([vals[i] - sd[i]], 0.005, 25.0)[0])
        hi = float(jit([vals[i] + sd[i]], 0.005, 25.0)[0])
        lo, hi = ordered_bounds(vj, lo, hi)
        if horizontal:
            row = {"series": "Yield", "x": r6(vj), "y": cat,
                   "y_err_lower": r6(lo), "y_err_upper": r6(hi)}
        else:
            row = {"series": "Yield", "x": cat, "y": r6(vj),
                   "y_err_lower": r6(lo), "y_err_upper": r6(hi)}
        row["error_bar_type"] = "SD"
        row["series_color"] = "#ff7f0e"
        rows.append(row)
    return rows


def make_bar():
    fig, ax = new_fig()
    ax.bar(_BAR_CATS, _BAR_VALS, yerr=_BAR_SD, color="#ff7f0e",
           capsize=5, width=0.6)
    ax.set_xlim(0.3, 5.7)
    ax.set_ylim(0, 27)
    ax.set_xlabel("Plot number")
    ax.set_ylabel("Yield (t/ha)")
    ax.grid(axis="y", alpha=0.3)
    px_of, w, h, png = finish(fig, ax, "bar")

    write_json("bar", plot_type="bar", orientation="vertical",
               image_bytes=png, w=w, h=h,
               axes=axes_block(px_of, (1, 5), (0, 25)),
               rows=_bar_rows(_BAR_VALS, _BAR_SD, horizontal=False),
               series=[{"key": "Yield", "color": "#ff7f0e"}])


def make_bar_horizontal():
    fig, ax = new_fig()
    ax.barh(_BAR_CATS, _BAR_VALS, xerr=_BAR_SD, color="#ff7f0e",
            capsize=5, height=0.6)
    ax.set_ylim(0.3, 5.7)
    ax.set_xlim(0, 27)
    ax.set_ylabel("Plot number")
    ax.set_xlabel("Yield (t/ha)")
    ax.grid(axis="x", alpha=0.3)
    px_of, w, h, png = finish(fig, ax, "bar_horizontal")

    write_json("bar_horizontal", plot_type="bar", orientation="horizontal",
               image_bytes=png, w=w, h=h,
               axes=axes_block(px_of, (0, 25), (1, 5)),
               rows=_bar_rows(_BAR_VALS, _BAR_SD, horizontal=True),
               series=[{"key": "Yield", "color": "#ff7f0e"}])


# ---------------------------------------------------------------------------
# 6/7. Box — vertical and horizontal variants, with an outlier
# ---------------------------------------------------------------------------

_BOX_STATS = [
    # cat, whislo, q1, med, q3, whishi, outliers
    (1, 3.1, 4.6, 5.4, 6.3, 7.9, [9.6]),
    (2, 4.0, 5.5, 6.5, 7.4, 8.8, []),
    (3, 2.2, 3.4, 4.1, 5.0, 6.4, [8.2]),
]


def _box_rows(horizontal):
    rows = []
    for cat, wlo, q1, med, q3, whi, fliers in _BOX_STATS:
        span = 10.0
        medj = float(jit([med], 0.004, span)[0])
        q1j = float(jit([q1], 0.004, span)[0])
        q3j = float(jit([q3], 0.004, span)[0])
        wloj = float(jit([wlo], 0.004, span)[0])
        whij = float(jit([whi], 0.004, span)[0])
        q1j, q3j = min(q1j, medj), max(q3j, medj)
        wloj, whij = ordered_bounds(medj, wloj, whij)
        base = {"series": f"Group {cat}",
                "box_q1": r6(q1j), "box_median": r6(medj), "box_q3": r6(q3j),
                "y_err_lower": r6(wloj), "y_err_upper": r6(whij),
                "status": "", "series_color": "#17becf"}
        if horizontal:
            base.update({"x": r6(medj), "y": cat})
        else:
            base.update({"x": cat, "y": r6(medj)})
        rows.append(base)
        for f in fliers:
            fj = float(jit([f], 0.004, span)[0])
            out = {"series": f"Group {cat}", "status": "outlier",
                   "series_color": "#17becf"}
            if horizontal:
                out.update({"x": r6(fj), "y": cat})
            else:
                out.update({"x": cat, "y": r6(fj)})
            rows.append(out)
    return rows


def _bxp_stats():
    return [dict(label=str(c), whislo=wl, q1=q1, med=m, q3=q3, whishi=wh,
                 fliers=fl)
            for c, wl, q1, m, q3, wh, fl in _BOX_STATS]


def make_box():
    fig, ax = new_fig()
    ax.bxp(_bxp_stats(), positions=[1, 2, 3], showfliers=True,
           boxprops=dict(color="#17becf"), medianprops=dict(color="#d62728"))
    ax.set_xlim(0.4, 3.6)
    ax.set_ylim(0, 11)
    ax.set_xlabel("Group")
    ax.set_ylabel("Measurement")
    ax.grid(axis="y", alpha=0.3)
    px_of, w, h, png = finish(fig, ax, "box")

    write_json("box", plot_type="box", orientation="vertical",
               image_bytes=png, w=w, h=h,
               axes=axes_block(px_of, (1, 3), (0, 10)),
               rows=_box_rows(horizontal=False),
               series=[{"key": f"Group {c}", "color": "#17becf"}
                       for c, *_ in _BOX_STATS])


def make_box_horizontal():
    fig, ax = new_fig()
    ax.bxp(_bxp_stats(), positions=[1, 2, 3], vert=False, showfliers=True,
           boxprops=dict(color="#17becf"), medianprops=dict(color="#d62728"))
    ax.set_ylim(0.4, 3.6)
    ax.set_xlim(0, 11)
    ax.set_ylabel("Group")
    ax.set_xlabel("Measurement")
    ax.grid(axis="x", alpha=0.3)
    px_of, w, h, png = finish(fig, ax, "box_horizontal")

    write_json("box_horizontal", plot_type="box", orientation="horizontal",
               image_bytes=png, w=w, h=h,
               axes=axes_block(px_of, (0, 10), (1, 3)),
               rows=_box_rows(horizontal=True),
               series=[{"key": f"Group {c}", "color": "#17becf"}
                       for c, *_ in _BOX_STATS])


# ---------------------------------------------------------------------------
# 8. Kaplan-Meier — two arms, censoring, at-risk counts
# ---------------------------------------------------------------------------

def make_km():
    t = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
    surv_a = np.array([1.0, 0.94, 0.86, 0.79, 0.70, 0.64, 0.57, 0.52, 0.48])
    surv_b = np.array([1.0, 0.90, 0.78, 0.66, 0.55, 0.47, 0.40, 0.34, 0.30])
    at_risk_a = np.array([120, 113, 103, 94, 83, 75, 66, 60, 54])
    at_risk_b = np.array([118, 106, 91, 77, 64, 54, 45, 38, 33])
    censored_a = {9, 18}
    censored_b = {12, 21}

    fig, ax = new_fig()
    for surv, c, lbl in ((surv_a, "#1f77b4", "Arm A"),
                         (surv_b, "#d62728", "Arm B")):
        ax.step(t, surv, where="post", color=c, label=lbl, lw=2)
        sd = 0.03
        ax.fill_between(t, surv - sd, surv + sd, step="post",
                        color=c, alpha=0.15)
    for tv in censored_a:
        ax.plot(tv, surv_a[t == tv], marker="|", ms=12, color="#1f77b4")
    for tv in censored_b:
        ax.plot(tv, surv_b[t == tv], marker="|", ms=12, color="#d62728")
    ax.set_xlim(-0.5, 25)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival probability")
    ax.legend()
    ax.grid(alpha=0.3)
    px_of, w, h, png = finish(fig, ax, "kaplan_meier")

    rows = []
    for surv, at_risk, cens, s, c in (
            (surv_a, at_risk_a, censored_a, "Arm A", "#1f77b4"),
            (surv_b, at_risk_b, censored_b, "Arm B", "#d62728")):
        for i, tv in enumerate(t):
            yj = float(np.clip(jit([surv[i]], 0.006, 1.0)[0], 0.0, 1.0))
            lo = float(np.clip(yj - 0.03, 0.0, 1.0))
            hi = float(np.clip(yj + 0.03, 0.0, 1.0))
            rows.append({"series": s, "x": int(tv), "y": r6(yj),
                         "y_err_lower": r6(lo), "y_err_upper": r6(hi),
                         "at_risk": int(at_risk[i]),
                         "status": "censored" if int(tv) in cens else "",
                         "error_bar_type": "Confidence",
                         "series_color": c})

    write_json("kaplan_meier", plot_type="kaplan_meier",
               orientation="vertical", image_bytes=png, w=w, h=h,
               axes=axes_block(px_of, (0, 24), (0, 1)),
               rows=rows,
               series=[{"key": "Arm A", "color": "#1f77b4"},
                       {"key": "Arm B", "color": "#d62728"}])


if __name__ == "__main__":
    make_scatter()
    make_time_series()
    make_forest()
    make_bar()
    make_bar_horizontal()
    make_box()
    make_box_horizontal()
    make_km()
    print("done")
