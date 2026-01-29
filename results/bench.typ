#import "@preview/lilaq:0.5.0" as lq

#set page(width: auto, height: auto)
#set text(font: "Rotis Sans Serif Std")

= Performance

#let data = csv("out.csv", row-type: dictionary)


#let extract(data, alloc, column) = {
  let filtered = data.filter(d => d.alloc == alloc)
  (
    filtered.map(d => int(d.x)),
    filtered.map(d => int(d.at(column))),
  )
}

#let base = lq.color.map.okabe-ito
#let cycle = (
  (color: base.at(0)),
  (color: base.at(1)),
  (color: base.at(0), stroke: (dash: "dashed")),
  (color: base.at(1), stroke: (dash: "dashed")),
)
#let draw-handle(color, stroke: (:), mark: none) = box(
  width: 2em,
  height: .7em,
  {
    let stroke = (paint: color, ..stroke)
    line(length: 100%, stroke: stroke)
    if mark != none {
      place(dx: 50%, (lq.marks.at(mark))((
        size: 5pt,
        fill: color,
        stroke: stroke,
      )))
    }
  },
)
#let legend = lq.legend(
  position: top + left,
  draw-handle(base.at(0)),
  [LLFree],
  draw-handle(base.at(1)),
  [LLC],
  [],
  [],
  draw-handle(black, mark: "."),
  [alloc],
  draw-handle(black, mark: ".", stroke: (dash: "dashed")),
  [free],
)

#show: lq.set-diagram(width: 8cm, height: 6cm)

#let data-o0 = data.filter(d => d.order == "0")
#lq.diagram(
  title: [Bulk Alloc - Order 0],
  xlabel: "Number of Cores",
  ylabel: "Allocation Time (ns)",
  ylim: (0, 100),
  cycle: cycle,
  legend: legend,
  lq.plot(..extract(data-o0, "LLFree", "get_avg"), label: "LLFree Get"),
  lq.plot(..extract(data-o0, "LLC", "get_avg"), label: "LLC Get"),
  lq.plot(..extract(data-o0, "LLFree", "put_avg"), label: "LLFree Put"),
  lq.plot(..extract(data-o0, "LLC", "put_avg"), label: "LLC Put"),
)

#let data-o9 = data.filter(d => d.order == "9")
#lq.diagram(
  title: [Bulk Alloc - Order 9],
  xlabel: "Number of Cores",
  ylabel: "Allocation Time (ns)",
  ylim: (0, 100),
  cycle: cycle,
  lq.plot(..extract(data-o9, "LLFree", "get_avg")),
  lq.plot(..extract(data-o9, "LLC", "get_avg")),
  lq.plot(..extract(data-o9, "LLFree", "put_avg")),
  lq.plot(..extract(data-o9, "LLC", "put_avg")),
)


== Fragmentation

#let data = (
  csv("frag.csv")
    .slice(1)
    .map(row => (
      i: int(row.at(0)),
      num_pages: int(row.at(1)),
      allocs: int(row.at(2)),
      huge_pages: row.slice(3).map(int),
    ))
)

#lq.diagram(
  title: [Free Huge Pages],
  xlim: (0, 100),
  xlabel: "Number of Iterations",
  ylabel: "Number of Free Huge Pages",
  lq.plot(
    data.map(d => d.i),
    data.map(d => d.huge_pages.filter(c => c == 512).len()),
    label: "100%",
    mark: none,
  ),
  lq.plot(
    data.map(d => d.i),
    data.map(d => d.huge_pages.filter(c => c < 512 and c > 496).len()),
    label: "95%-99%",
    mark: none,
  ),
)

#let div = 16
#let mesh = lq.colormesh(
  data.map(d => d.i),
  range(int(data.first().huge_pages.len() / div)),
  (x, y) => (
    data.at(x).huge_pages.slice(y * div, (y + 1) * div).sum() / (div * 512)
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
