/**
 * Extracts the bracketed extras-group name from a VideoAnnotator install hint,
 * e.g. "pip install videoannotator[face]" -> "face".
 *
 * Returns null when the hint doesn't match the expected shape, so callers can
 * fall back to showing the locked pipeline without an Install action rather
 * than calling the install endpoint with a bad/guessed group name.
 */
export function extraNameFromInstallHint(hint: string | undefined | null): string | null {
  if (!hint) return null;
  const match = hint.match(/\[([a-zA-Z0-9_-]+)\]/);
  return match ? match[1] : null;
}
