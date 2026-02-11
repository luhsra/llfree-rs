#import "@preview/lilaq:0.5.0" as lq

/// Group a list of dictionaries by specified keys, creating a nested dictionary structure based on the unique values of those keys.
///
/// - data (dictionary): The input list of dictionaries to be grouped.
/// - keys (string | int): The keys or indices to group the dictionaries by. The grouping will be performed in the order of the keys provided.
/// -> dictionary
#let group-by(data, ..keys) = {
  let key-list = keys.pos()
  if key-list.len() == 0 {
    data
  } else {
    let (first-key, ..rest-keys) = key-list
    let grouped = data.fold((:), (group, entry) => {
      let k = entry.remove(first-key)
      if k in group {
        group.at(k).push(entry)
      } else {
        group.insert(k, (entry,))
      }
      group
    })
    if rest-keys.len() > 0 {
      grouped
        .pairs()
        .fold((:), (result, (k, v)) => {
          result.insert(k, group-by(v, ..rest-keys))
          result
        })
    } else {
      grouped
    }
  }
}

/// Recursively apply a mapper function to the values of a dictionary at a specified level of nesting.
///
/// - dict (dictionary): The input dictionary, where the values are more dictionaries up to the specified level.
/// - level (int): The depth level to apply the mapper function.
/// - mapper (function): The function to apply to the values at the specified level.
/// -> dictionary
#let dict-map(dict, level: 0, mapper) = {
  if level == 0 {
    dict.pairs().map(((k, v)) => (k, mapper(v))).to-dict()
  } else {
    dict
      .pairs()
      .map(((k, v)) => (k, dict-map(v, level: level - 1, mapper)))
      .to-dict()
  }
}

/// Recursively extract values from a nested dictionary structure at a specified level of nesting, optionally applying a mapper function to the extracted values.
///
/// - dict (dictionary): The input dictionary, where the values are more dictionaries up to the specified level.
/// - level (int): The depth level to extract the values from.
/// - mapper (function | none): An optional function to apply to the extracted values.
/// -> array
#let dict-values(dict, level: 0, mapper: none) = {
  if level == 0 {
    if mapper != none {
      dict.values().map(mapper)
    } else {
      dict.values()
    }
  } else {
    dict.values().map(v => dict-values(v, level: level - 1, mapper: mapper))
  }
}

#let stddev(values) = {
  let avg = values.sum() / values.len()
  calc.sqrt(values.map(v => calc.pow(v - avg, 2)).sum() / (values.len() - 1))
}
#let stats(values) = (
  min: values.reduce(calc.min),
  max: values.reduce(calc.max),
  avg: values.sum() / values.len(),
  stddev: stddev(values),
)

/// Extract columns from a dict of dicts and return them as separate lists
/// with the keys as the first list.
///
/// - dict (dictionary): The input dictionary, where the values are floats, dictionaries or arrays.
/// - columns (string | int): The keys or indices of the columns to extract from the inner dictionaries or arrays or none if the values are floats.
/// -> dictionary
#let extract(dict, ..columns) = {
  let keys = dict.keys().map(float)
  let values = if columns.pos().len() > 0 {
    columns
      .pos()
      .map(col => (col, dict.values().map(r => r.map(v => float(v.at(col))))))
      .to-dict()
  } else {
    (values: dict.values().map(float))
  }
  (keys: keys, ..values)
}

/// Colormaps from seaborn (https://seaborn.pydata.org/tutorial/color_palettes.html)
#let colormap = (
  spectral: (
    "#b41947",
    "#c9314c",
    "#da464d",
    "#e75948",
    "#f46d43",
    "#f8864f",
    "#fba05b",
    "#fdb768",
    "#fecc7b",
    "#fee08b",
    "#feec9f",
    "#fff8b4",
    "#fafdb7",
    "#f0f9a7",
    "#e6f598",
    "#cfec9d",
    "#b5e1a2",
    "#9cd7a4",
    "#81cda5",
    "#66c2a5",
    "#50a9af",
    "#3b92b9",
    "#3b7cb7",
    "#4d65ad",
  ).map(rgb),
  colorblind: (
    "#0173b2",
    "#de8f05",
    "#029e73",
    "#d55e00",
    "#cc78bc",
    "#ca9161",
    "#fbafe4",
    "#949494",
    "#ece133",
    "#56b4e9",
  ).map(rgb),
  colorblind6: (
    "#0173b2",
    "#029e73",
    "#d55e00",
    "#cc78bc",
    "#ece133",
    "#56b4e9",
  ).map(rgb),
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
        stroke: color,
      )))
    }
  },
)
