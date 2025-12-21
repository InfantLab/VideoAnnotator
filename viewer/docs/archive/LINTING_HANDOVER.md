# Linting Handover (Dec 2025)

This repo is intentionally **strict** about lint/type safety. The goal is: keep `npm run lint` at **0 errors / 0 warnings** without weakening rules.

## What was going wrong

The remaining lint noise (after TypeScript correctness issues were fixed) mostly came from two ESLint rules:

### 1) `react-refresh/only-export-components`
**Symptom:** Warnings like “Fast refresh only works when a file only exports components”.

**Root cause:** In React Fast Refresh mode, ESLint flags modules that export a React component **and** also export any non-component value (constants, helpers, variant builders, hooks, types-as-values, etc.).

**Typical offenders:**
- UI component modules exporting `*Variants` (CVA) alongside components
- Context modules exporting both provider components and helper hooks/constants
- Utility modules that happen to export a component plus other helpers

**Recommended fix (preferred): split exports by responsibility**
- Keep component files exporting **components only**.
- Move non-component exports into sibling modules.

Patterns used in this repo:
- `*-variants.ts` for CVA variant builders
  - Example: [src/components/ui/button-variants.ts](../src/components/ui/button-variants.ts)
  - Component imports it and does not re-export it: [src/components/ui/button.tsx](../src/components/ui/button.tsx)
- `*Settings.ts` / `*settings.ts` for configuration types/defaults
  - Example: [src/components/openface3Settings.ts](../src/components/openface3Settings.ts)
- `*Provider.tsx` for provider components, while `*Context.tsx` exports context + hooks only
  - Example: [src/contexts/PipelineContext.tsx](../src/contexts/PipelineContext.tsx) + [src/contexts/PipelineProvider.tsx](../src/contexts/PipelineProvider.tsx)
  - Example: [src/contexts/ServerCapabilitiesContext.tsx](../src/contexts/ServerCapabilitiesContext.tsx) + [src/contexts/ServerCapabilitiesProvider.tsx](../src/contexts/ServerCapabilitiesProvider.tsx)

### 2) `react-hooks/exhaustive-deps`
**Symptom:** Warnings about missing or unnecessary dependencies in `useEffect`/`useCallback`.

**Root cause:** Effects/callbacks captured values that weren’t declared as dependencies, or included unstable derived values (new arrays/objects each render), causing churn.

**Recommended fix (preferred): make dependencies correct, stabilize inputs**
- Do not disable the rule.
- Prefer:
  - moving helper functions above effects and wrapping them in `useCallback`
  - using `useMemo` for derived arrays/objects used in deps
  - ensuring cleanup functions are included via stable callbacks

## Recommended workflow to fix lint cleanly

1) Run lint and capture exact offenders
- `npm run lint`

2) Fix error-level issues first
Common error-level patterns we’ve seen:
- `@typescript-eslint/no-explicit-any`: replace `any` with concrete types, unions, generics, or `unknown` + narrowing.
- “unsafe” property access: validate external data at boundaries (Zod helpers in [src/lib/validation.ts](../src/lib/validation.ts)).

3) Fix warning-level issues without relaxing rules
- For `react-refresh/only-export-components`: split modules (see patterns above).
- For hooks deps: correct dependency lists, stabilize derived values.

4) Re-run lint until clean
- `npm run lint`

5) Confirm typecheck
- `npx tsc --noEmit`

## Quick decision tree

### If ESLint complains about Fast Refresh exports
- Does the file export a component AND something else?
  - Yes → move the “something else” to another module and import it back.
  - No → the rule might be triggered by an indirect export; check re-exports.

### If ESLint complains about hook deps
- Missing dep?
  - Add it, then make sure the referenced function/value is stable (wrap in `useCallback` / `useMemo`).
- “Unnecessary dep”?
  - Remove it and re-check that the callback/effect still observes the correct values.

## Common anti-patterns (avoid)
- Exporting CVA variant builders from a component module.
- Exporting a provider component and context hooks from the same file.
- Silencing `exhaustive-deps` with eslint-disable.
- Using derived arrays/objects inline inside deps (e.g. `const x = [a,b]` then `useEffect(...,[x])`).

## Notes
- The strict lint posture is intentional: it keeps the codebase predictable and prevents regressions during fast iteration.
- If you must intentionally break a rule for a narrow case, prefer a refactor first; only then discuss a scoped eslint disable with justification.
