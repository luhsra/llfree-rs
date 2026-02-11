#import "util.typ": *

#import "@preview/lilaq:0.5.0" as lq

#show: lq.set-diagram(width: 6cm, height: 4cm)

#set page(width: auto, height: auto)
#set text(font: "Rotis Sans Serif Std")

#let data = csv("bulk.csv", row-type: dictionary)
#let data = group-by(data, "alloc", "order", "x")

#let (first, second, ..) = colormap.colorblind
#let cycle = (
  (color: first),
  (color: second),
  (color: first, stroke: (dash: "dashed")),
  (color: second, stroke: (dash: "dashed")),
)

#let legend = lq.legend(
  position: top + left,
  draw-handle(first),
  [LLFree],
  draw-handle(second),
  [LLC],
  [],
  [],
  draw-handle(black, mark: "."),
  [alloc],
  draw-handle(black, mark: ".", stroke: (dash: "dashed")),
  [free],
)

#let extracted = dict-map(data, level: 1, v => {
  let (keys, ..values) = extract(v, "get_avg", "put_avg")
  (keys: keys, ..dict-map(values, s => s.map(stats)))
})

#let ymax = (
  dict-values(extracted, level: 1, mapper: v => v.values().slice(1))
    .flatten()
    .map(s => s.max)
    .reduce(calc.max)
    * 1.15
)

#let plot-args(stats, key) = arguments(
  stats.keys,
  stats.at(key).map(s => s.avg),
  yerr: stats.at(key).map(s => s.stddev),
)

#lq.diagram(
  title: [Bulk Alloc - Order 0],
  xlabel: "Number of Cores",
  ylabel: "Allocation Time (ns)",
  ylim: (0, ymax),
  cycle: cycle,
  legend: legend,
  lq.plot(..plot-args(extracted.LLFree.at("0"), "get_avg")),
  lq.plot(..plot-args(extracted.LLC.at("0"), "get_avg")),
  lq.plot(..plot-args(extracted.LLFree.at("0"), "put_avg")),
  lq.plot(..plot-args(extracted.LLC.at("0"), "put_avg")),
)
#lq.diagram(
  title: [Bulk Alloc - Order 9],
  xlabel: "Number of Cores",
  // ylabel: "Allocation Time (ns)",
  yaxis: (format-ticks: none),
  ylim: (0, ymax),
  cycle: cycle,
  lq.plot(..plot-args(extracted.LLFree.at("9"), "get_avg")),
  lq.plot(..plot-args(extracted.LLC.at("9"), "get_avg")),
  lq.plot(..plot-args(extracted.LLFree.at("9"), "put_avg")),
  lq.plot(..plot-args(extracted.LLC.at("9"), "put_avg")),
)
