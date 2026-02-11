#import "@preview/lilaq:0.5.0" as lq

#import "util.typ": *

#set page(width: auto, height: auto)
#set text(font: "Rotis Sans Serif Std")

#let data = (
  read("frag-l.txt")
    .trim()
    .split("\n")
    .map(row => (
      row.codepoints().map(c => calc.clamp(int(c) * 64 - 32, 0, 512))
    ))
)

#lq.diagram(
  title: [Free Huge Pages],
  xlim: (0, auto),
  xlabel: "Number of Iterations",
  ylabel: "Number of Huge Pages",
  lq.plot(
    range(int(data.len())),
    data.map(d => d.filter(c => c == 512).len()),
    label: "100%",
    mark: none,
  ),
  lq.plot(
    range(int(data.len())),
    data.map(d => d.filter(c => c >= 512 - 32 and c < 512).len()),
    label: str(calc.round((512 - 32) / 512 * 100)) + "%-99%",
    mark: none,
  ),
)

#let div = 4
#let mesh = lq.colormesh(
  range(data.len()),
  range(int(data.first().len() / div)),
  (x, y) => (
    data.at(x).slice(y * div, (y + 1) * div).sum() / div
  ),
  map: (
    colormap.spectral.first().darken(20%),
    ..colormap.spectral.slice(1, -1),
    colormap.spectral.last().darken(20%),
  ),
)
#lq.diagram(
  xlim: (0, 100),
  xlabel: "Number of Iterations",
  ylabel: "Huge Page",
  mesh,
)
#lq.colorbar(mesh, label: [Fraction of Used Pages])
