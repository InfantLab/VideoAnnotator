// Hues from each value's rank among the distinct values, not from its characters: a
// first-letter hash gave SPEAKER_00 and SPEAKER_01 the same colour. Numeric-aware sort
// keeps the order stable across files, so the overlay and timeline agree.
const GOLDEN_ANGLE = 137.508;

export const categoryHues = (values: Iterable<string>): Map<string, number> => {
  const distinct = [...new Set(values)].sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true }),
  );
  return new Map(distinct.map((value, i) => [value, (i * GOLDEN_ANGLE) % 360]));
};
