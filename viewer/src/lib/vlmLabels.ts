/** How the viewer reads a VLM label against touch ground truth. */

/** A label that says touch is happening (not an error, a "no", or empty). */
export const isPositiveLabel = (label: string): boolean =>
  !label.startsWith('ERROR') && label !== 'NO_TOUCH' && label !== 'NO' && label !== 'EMPTY';
