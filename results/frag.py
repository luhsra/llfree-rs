import marimo

__generated_with = "0.23.6"
app = marimo.App(auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import colors
    from typing import Tuple
    import numpy as np

    sns.set_style("whitegrid")
    sns.set_context("poster", font_scale=0.75)
    sns.set_palette("colorblind")

    def parse_fragout(file: Path) -> pd.DataFrame:
        out = []
        with file.open() as f:
            for line in f:
                row = np.zeros(len(line), dtype=np.int8)
                for i, char in enumerate(line[:-1]):
                    row[i] = int(char)
                out.append(row)
        out = np.array(out)
        return pd.DataFrame(out)

    return Path, cm, parse_fragout, plt, sns


@app.cell
def _(Path, cm, parse_fragout, plt, sns):
    _data = parse_fragout(Path('frag-c.txt'))
    _per_huge = _data[[*_data.columns[2:]]]
    _buckets = _per_huge.apply(lambda d: d.value_counts(), axis=1).fillna(0)
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(10)
    _fix.set_figheight(6)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _buckets.plot.area(ax=_ax, xlim=(0, 100), xlabel='iteration (N*0.05 reallocations)', ylabel='huge pages', yticks=[], legend=False, colormap=_cmap)
    plt.colorbar(cm.ScalarMappable(cmap=_cmap), ax=_ax, extend='both', label='free pages per huge page')
    _ax
    return


@app.cell
def _(Path, cm, parse_fragout, plt, sns):
    _data = parse_fragout(Path('frag-r.txt'))
    _per_huge = _data[[*_data.columns[2:]]]
    _buckets = _per_huge.apply(lambda d: d.value_counts(), axis=1).fillna(0)
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(10)
    _fix.set_figheight(6)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _buckets.plot.area(ax=_ax, xlim=(0, 100), xlabel='iteration (N*0.05 reallocations)', ylabel='huge pages', yticks=[], legend=False, colormap=_cmap)
    plt.colorbar(cm.ScalarMappable(cmap=_cmap), ax=_ax, extend='both', label='free pages per huge page')
    _ax
    return


@app.cell
def _(Path, cm, parse_fragout, plt, sns):
    _data = parse_fragout(Path('frag-r1.txt'))
    _per_huge = _data[[*_data.columns[2:]]]
    _buckets = _per_huge.apply(lambda d: d.value_counts(), axis=1).fillna(0)
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(10)
    _fix.set_figheight(6)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _buckets.plot.area(ax=_ax, xlim=(0, 100), xlabel='iteration (N*0.05 reallocations)', ylabel='huge pages', yticks=[], legend=False, colormap=_cmap)
    plt.colorbar(cm.ScalarMappable(cmap=_cmap), ax=_ax, extend='both', label='free pages per huge page')
    _ax
    return


@app.cell
def _(Path, parse_fragout, plt, sns):
    # Heatmap Array
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(20)
    _fix.set_figheight(12)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _data = parse_fragout(Path('frag-c.txt'))
    _per_huge = _data.T
    _plot = sns.heatmap(_per_huge, ax=_ax, cmap=_cmap, yticklabels=2 * 1024, xticklabels=10)
    _plot.set(ylabel='Huge pages')
    _plot.set(xlabel='iteration')
    return


@app.cell
def _(Path, parse_fragout, plt, sns):
    # Heatmap Array
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(20)
    _fix.set_figheight(12)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _data = parse_fragout(Path('frag-r.txt'))
    _per_huge = _data.T
    _plot = sns.heatmap(_per_huge, ax=_ax, cmap=_cmap, yticklabels=2 * 1024, xticklabels=10)
    _plot.set(ylabel='Huge pages')
    _plot.set(xlabel='iteration')
    return


@app.cell
def _(Path, parse_fragout, plt, sns):
    # Heatmap Array
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(20)
    _fix.set_figheight(12)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _data = parse_fragout(Path('stress-r.txt'))
    _per_huge = _data.T
    print(_per_huge.size, _per_huge.columns.size, _per_huge.size / _per_huge.columns.size)
    _huge_pages = _per_huge.size // _per_huge.columns.size
    _plot = sns.heatmap(_per_huge, ax=_ax, cmap=_cmap, yticklabels=2 * 1024, xticklabels=10)
    _plot.set(ylabel='Huge pages')
    # plot.hlines(list(range(0, huge_pages, 32)), 0, per_huge.columns.size, colors="black", linewidth=0.4)
    _plot.set(xlabel='iteration')
    return


@app.cell
def _(Path, parse_fragout, plt, sns):
    # Heatmap Array
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(20)
    _fix.set_figheight(12)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _data = parse_fragout(Path('stress-r1.txt'))
    _per_huge = _data.T
    print(_per_huge.size, _per_huge.columns.size, _per_huge.size / _per_huge.columns.size)
    _huge_pages = _per_huge.size // _per_huge.columns.size
    _plot = sns.heatmap(_per_huge, ax=_ax, cmap=_cmap, yticklabels=2 * 1024, xticklabels=10)
    _plot.set(ylabel='Huge pages')
    # plot.hlines(list(range(0, huge_pages, 32)), 0, per_huge.columns.size, colors="black", linewidth=0.4)
    _plot.set(xlabel='iteration')
    return


@app.cell
def _(Path, parse_fragout, plt, sns):
    # Heatmap Array
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(20)
    _fix.set_figheight(12)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _data = parse_fragout(Path('stress-c.txt'))
    _per_huge = _data.T
    print(_per_huge.size, _per_huge.columns.size, _per_huge.size / _per_huge.columns.size)
    _huge_pages = _per_huge.size // _per_huge.columns.size
    _plot = sns.heatmap(_per_huge, ax=_ax, cmap=_cmap, yticklabels=2 * 1024, xticklabels=10)
    _plot.set(ylabel='Huge pages')
    # plot.hlines(list(range(0, huge_pages, 32)), 0, per_huge.columns.size, colors="black", linewidth=0.4)
    _plot.set(xlabel='iteration')
    return


@app.cell(hide_code=True)
def _(Path, cm, parse_fragout, plt, sns):
    _data = parse_fragout(Path('stress-r.txt'))
    _per_huge = _data[[*_data.columns[2:]]]
    _buckets = _per_huge.apply(lambda d: d.value_counts(), axis=1).fillna(0)
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(10)
    _fix.set_figheight(6)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _buckets.plot.area(ax=_ax, xlim=(0, None), xlabel='iteration (N*0.05 reallocations)', ylabel='huge pages', yticks=[], legend=False, colormap=_cmap)
    plt.colorbar(cm.ScalarMappable(cmap=_cmap), ax=_ax, extend='both', label='free pages per huge page')
    _ax
    return


@app.cell
def _(Path, cm, parse_fragout, plt, sns):
    _data = parse_fragout(Path('stress-r1.txt'))
    _per_huge = _data[[*_data.columns[2:]]]
    _buckets = _per_huge.apply(lambda d: d.value_counts(), axis=1).fillna(0)
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(10)
    _fix.set_figheight(6)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _buckets.plot.area(ax=_ax, xlim=(0, None), xlabel='iteration (N*0.05 reallocations)', ylabel='huge pages', yticks=[], legend=False, colormap=_cmap)
    plt.colorbar(cm.ScalarMappable(cmap=_cmap), ax=_ax, extend='both', label='free pages per huge page')
    _ax
    return


@app.cell
def _(Path, cm, parse_fragout, plt, sns):
    _data = parse_fragout(Path('stress-c.txt'))
    _per_huge = _data[[*_data.columns[2:]]]
    _buckets = _per_huge.apply(lambda d: d.value_counts(), axis=1).fillna(0)
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(10)
    _fix.set_figheight(6)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _buckets.plot.area(ax=_ax, xlim=(0, None), xlabel='seconds', ylabel='huge pages', yticks=[], legend=False, colormap=_cmap)
    plt.colorbar(cm.ScalarMappable(cmap=_cmap), ax=_ax, extend='both', label='free pages per huge page')
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Replay
    """)
    return


@app.cell
def _(Path, cm, parse_fragout, plt, sns):
    _data = parse_fragout(Path('replay-clang-c.csv'))
    _per_huge = _data[[*_data.columns[2:]]]
    _buckets = _per_huge.apply(lambda d: d.value_counts(), axis=1).fillna(0)
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(10)
    _fix.set_figheight(6)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _buckets.plot.area(ax=_ax, xlim=(0, None), xlabel='iter', ylabel='huge pages', yticks=[], legend=False, colormap=_cmap)
    plt.colorbar(cm.ScalarMappable(cmap=_cmap), ax=_ax, extend='both', label='free pages per huge page')
    _fix
    return


@app.cell
def _(Path, cm, parse_fragout, plt, sns):
    _data = parse_fragout(Path('replay-clang-r.csv'))
    _per_huge = _data[[*_data.columns[2:]]]
    _buckets = _per_huge.apply(lambda d: d.value_counts(), axis=1).fillna(0)
    _fix, _ax = plt.subplots()
    _fix.set_figwidth(10)
    _fix.set_figheight(6)
    _fix.set_facecolor('white')
    _cmap = sns.color_palette('Spectral', as_cmap=True, n_colors=10)
    _buckets.plot.area(ax=_ax, xlim=(0, None), xlabel='iter', ylabel='huge pages', yticks=[], legend=False, colormap=_cmap)
    plt.colorbar(cm.ScalarMappable(cmap=_cmap), ax=_ax, extend='both', label='free pages per huge page')
    _fix
    return


if __name__ == "__main__":
    app.run()
