#import "@preview/lilaq:0.5.0" as lq

#set page(width: auto, height: auto)
#set text(font: "Rotis Sans Serif Std")

== Fragmentation

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
  ylabel: "Number of Free Huge Pages",
  lq.plot(
    range(int(data.len())),
    data.map(d => d.filter(c => c == 512).len()),
    label: "100%",
    mark: none,
  ),
  lq.plot(
    range(int(data.len())),
    data.map(d => d.filter(c => c == 512 - 32).len()),
    label: "85%-99%",
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
  map: lq.color.map.roma,
)
#lq.diagram(
  xlim: (0, 100),
  xlabel: "Number of Iterations",
  ylabel: "Huge Page",
  mesh,
)
#lq.colorbar(mesh, label: [Fraction of Used Pages])
