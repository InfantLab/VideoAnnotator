# Contract: Admin Authentication & Restart-Required Signal

## Admin authentication

Applies to both endpoints in [extras-install-endpoints.md](extras-install-endpoints.md).

- No credentials / invalid API key → `401`, same envelope this API already uses for missing/invalid
  auth (`validate_required_api_key`'s existing behavior).
- Valid credentials, but the authenticated user's `is_admin` is false → `403`:

```json
{
  "detail": "Administrator privileges required for this action."
}
```

- Valid admin credentials → request proceeds to the extras-name validation described in
  extras-install-endpoints.md.

This ordering matters and is tested (spec.md User Story 3, SC-002/SC-003): auth is checked before
extras-name validation, and extras-name validation happens before any subprocess is invoked — an
unauthenticated caller naming an invalid extras group gets `401`, not `422`.

## Restart-required signal semantics

- **Starts false** on server process startup, unconditionally — including if the database contains
  previously-completed install jobs from an earlier process lifetime. It answers "has a restart
  happened *since* the last completed install," not "has this extras group ever been installed."
- **Becomes true** the instant any `ExtrasInstallJob` transitions to `completed`, in this process.
- **Stays true** across further installs (does not reset because a *different* install started or
  even failed) until the process actually restarts.
- **Never becomes true** for a job that only reaches `failed` — a failed install has, by definition,
  not changed the environment in a way that needs a restart to take effect (or if it partially did,
  the environment is already in a state the failure output should guide the admin to fix and retry,
  not a state a restart would resolve).
- Exposed identically whether read from a specific job's status (`restart_required` field on that
  job) or from the general pipeline-listing endpoint's top-level `restart_required` — both reflect
  the same single in-process flag, not independent computations that could disagree.
