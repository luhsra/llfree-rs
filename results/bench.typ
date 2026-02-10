#import "@preview/lilaq:0.5.0" as lq

#show: lq.set-diagram(width: 7cm, height: 4cm)

#set page(width: auto, height: auto)
#set text(font: "Rotis Sans Serif Std")

#let data = csv("bulk.csv", row-type: dictionary)

#let iter = data.map(d => int(d.iteration)).reduce(calc.max) + 1

#let group-by(data, ..keys) = {
  let key-list = keys.pos()
  if key-list.len() == 0 {
    data
  } else {
    let (first-key, ..rest-keys) = key-list
    let grouped = data.fold((:), (group, entry) => {
      let k = entry.at(first-key)
      if k in group {
        group.at(k).push(entry)
      } else {
        group.insert(k, (entry,))
      }
      group
    })
    if rest-keys.len() > 0 {
      grouped.pairs().fold((:), (result, pair) => {
        let (k, v) = pair
        result.insert(k, group-by(v, ..rest-keys))
        result
      })
    } else {
      grouped
    }
  }
}

#let chunks = (
  group-by(data, "alloc", "order", "x")
)

#let stddev(values) = {
  let avg = values.sum() / values.len()
  calc.sqrt(values.map(v => calc.pow(v - avg, 2)).sum() / (values.len() - 1))
}

#let extract(data, alloc, column) = {
  let chunks = data.filter(d => d.alloc == alloc).chunks(iter)
  let values = chunks.map(d => d.map(r => int(r.at(column))))
  arguments(
    chunks.map(d => d.map(r => int(r.x)).sum() / iter),
    values.map(d => d.sum() / iter),
    yerr: values.map(d => stddev(d)),
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

#let ymax = (
  calc.max(
    data.map(d => int(d.get_avg)).reduce(calc.max),
    data.map(d => int(d.put_avg)).reduce(calc.max),
  )
    * 1.1
)

#let data-o0 = data.filter(d => d.order == "0")
#lq.diagram(
  title: [Bulk Alloc - Order 0],
  xlabel: "Number of Cores",
  ylabel: "Allocation Time (ns)",
  ylim: (0, ymax),
  cycle: cycle,
  legend: legend,
  lq.plot(..extract(data-o0, "LLFree", "get_avg")),
  lq.plot(..extract(data-o0, "LLC", "get_avg")),
  lq.plot(..extract(data-o0, "LLFree", "put_avg")),
  lq.plot(..extract(data-o0, "LLC", "put_avg")),
)

#let data-o9 = data.filter(d => d.order == "9")
#lq.diagram(
  title: [Bulk Alloc - Order 9],
  xlabel: "Number of Cores",
  ylabel: "Allocation Time (ns)",
  ylim: (0, ymax),
  cycle: cycle,
  lq.plot(..extract(data-o9, "LLFree", "get_avg")),
  lq.plot(..extract(data-o9, "LLC", "get_avg")),
  lq.plot(..extract(data-o9, "LLFree", "put_avg")),
  lq.plot(..extract(data-o9, "LLC", "put_avg")),
)
