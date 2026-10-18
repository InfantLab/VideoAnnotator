# Feature Specification: Hugging Face Token Setup

**Feature Branch**: `031-huggingface-token-setup` (work lands on `1.6-dev`)
**Created**: 2026-10-17
**Status**: Draft
**Input**: User description: "The note about setting the Hugging Face token is not very
user-friendly: 'A Hugging Face access token (to download the pyannote models) isn't set. Set
HUGGINGFACE_TOKEN in the server's environment (e.g. the container env or .env) and restart the
server. Get a token'. Can we do better with the process here?" Agreed design (2026-10-17): a guided
panel on the pipeline's card (create a token, accept the licence, paste it), saved on the server and
used at once, admin only, never returned to the viewer; a Settings row; a CLI equivalent.

## Context

Speaker turns (and the deprecated audio pipeline) download pyannote models, which Hugging Face only
gives to an account that has accepted their licence, using that account's token. The server already
checks a lot (spec 011): whether a token is set, whether Hugging Face accepts it, which account it
belongs to, and whether that account accepted each licence. What a researcher is asked to do with
that is the hard part: find the server's environment (a container's, or a `.env` they've never
seen), add a variable, restart. Most researchers stop there.

Pipelines read the token when a job starts, not when the server starts, so a token given to the
running server can be used straight away.

### What the researcher sees

On the pipeline's card, instead of the one-line note:

```
Speaker turns needs a free Hugging Face token (to download its models).
 1. Create a Hugging Face account and a "Read" token      [Open huggingface.co ↗]
 2. Accept the model's licence, signed in as that account  [pyannote/speaker-diarization-3.1 ↗]  ✗ not yet
 3. Paste the token here  [hf_••••••••••••]  [Save and check]
    ✓ Token works (account: caspar) · ✓ licence accepted · ready to run
```

Not an admin: "Ask whoever runs this VideoAnnotator to add a Hugging Face token."

In Settings: "Hugging Face token: set (account caspar, saved on this server) [Change] [Remove]".

From a terminal: `videoannotator settings huggingface-token` asks for the token (not shown as
typed), checks it, saves it.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Add the token from the pipeline's card (Priority: P1)

An admin sees that speaker turns needs a token. They follow steps 1 and 2 from the card's links,
paste the token, and press "Save and check". The card says the token works, names the account, and
shows each licence as accepted or not; once all are accepted, the pipeline is ready, without a
restart.

**Why this priority**: it replaces the step most researchers can't do.

**Independent Test**: on a server without a token, set one from the card and run speaker turns on
the demo clip, with no restart.

**Acceptance Scenarios**:

1. **Given** no token, **When** an admin opens the card, **Then** the three steps are shown, with
   the token page and each licence's model page linked.
2. **Given** a pasted token Hugging Face accepts, **When** "Save and check" is pressed, **Then** it
   is saved, the card names the account, and the readiness is re-checked at once.
3. **Given** a pasted token Hugging Face rejects, **When** saved, **Then** nothing is saved and the
   card says the token was rejected (expired, revoked or mis-pasted).
4. **Given** Hugging Face can't be reached, **When** saved, **Then** the token is saved and the card
   says it couldn't be checked yet.
5. **Given** a licence not yet accepted, **When** the admin accepts it and presses "Check again",
   **Then** its ✗ becomes ✓.
6. **Given** a non-admin, **When** they open the card, **Then** they see who to ask, and no field.

---

### User Story 2 - See, change or remove the token (Priority: P2)

In Settings, anyone signed in sees whether a token is set, where it comes from (saved on this
server, or the server's environment) and its account. An admin can change or remove a saved token.

**Why this priority**: tokens expire and get revoked; a lab needs to swap one without hunting for it.

**Independent Test**: set, change and remove the token from Settings; readiness follows each change.

**Acceptance Scenarios**:

1. **Given** a saved token, **When** Settings opens, **Then** it shows "set", "saved on this
   server" and the account, never the token.
2. **Given** a token in the server's environment, **When** Settings opens, **Then** it says so and
   that the environment's token is the one used; saving one in the viewer doesn't override it.
3. **Given** a saved token, **When** an admin removes it, **Then** it is deleted and the pipeline
   asks for a token again.

---

### User Story 3 - From a terminal (Priority: P3)

`videoannotator settings huggingface-token` asks for the token without echoing it, checks it with
Hugging Face, and saves it where the server reads it; `--remove` deletes it; `--status` shows what
Settings shows.

**Why this priority**: agents and headless servers (roadmap principle "agents are users too").

**Independent Test**: set the token from the CLI with no server running; start the server; the
pipeline is ready.

**Acceptance Scenarios**:

1. **Given** the command, **When** a token is typed, **Then** it is not shown, is checked, and saved.
2. **Given** `--status`, **When** run, **Then** it prints set/not set, source and account.
3. **Given** `--remove`, **When** run, **Then** the saved token is deleted.

---

### Edge Cases

- A token in both the environment and saved: the environment's wins (existing setups unchanged);
  Settings says so.
- The saved file is unreadable or empty: treated as not set, with a warning in the server log.
- Leading or trailing spaces in the pasted token are trimmed.
- A token that works but whose account hasn't accepted a licence: saved; the licence shows ✗.
- Hugging Face is down: saved, "couldn't check"; the next readiness check confirms it.
- The token never appears in responses, logs, provenance or exports.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The server MUST resolve the Hugging Face token from, in order: `HUGGINGFACE_TOKEN`,
  `HF_AUTH_TOKEN`, `HF_TOKEN` (as today), then a token saved on the server.
- **FR-002**: A saved token MUST be stored in a file next to the database, readable only by the
  server's user, so it survives container recreation and never leaves the machine.
- **FR-003**: An admin-only endpoint MUST save a token after checking it with Hugging Face: rejected
  tokens are not saved; an unreachable Hugging Face saves it and says it wasn't checked.
- **FR-004**: An admin-only endpoint MUST remove the saved token.
- **FR-005**: An endpoint for any signed-in user MUST report whether a token is set, its source
  (environment or saved), the environment variable when relevant, and the account name, never the
  token.
- **FR-006**: A saved token MUST be used by the next readiness check and the next job, without a
  restart.
- **FR-007**: The readiness message for a missing or rejected token MUST point to the card and
  Settings first, and the environment variable second.
- **FR-008**: The viewer MUST show the three-step panel on a pipeline that needs the token, with the
  token page and each pending licence's model page linked and each licence's state shown; the field
  only for admins.
- **FR-009**: The viewer's Settings MUST show the token's status, and let an admin change or remove
  a saved token.
- **FR-010**: The CLI MUST offer `videoannotator settings huggingface-token` (prompt without echo),
  with `--remove` and `--status`, working without a server.
- **FR-011**: The token MUST NOT appear in any API response, log line, provenance record or export.

### Key Entities

- **Saved token**: one value, in a file next to the database, mode 0600.
- **Token status**: set, source (`environment` | `saved` | none), environment variable name,
  account, check result (`ok` | `invalid` | `unknown`).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An admin goes from "needs a token" to running speaker turns without touching a file,
  an environment variable or a restart.
- **SC-002**: No API response, log line or export contains the token (checked by a test that
  searches them for it).
- **SC-003**: Existing setups with `HUGGINGFACE_TOKEN` in the environment behave exactly as before.

## Assumptions

- Admin-only, like installing pipelines (spec 005); the default single-user setup's key is admin.
- Only the Hugging Face token: other secrets (none today) can follow the same pattern later.
- Plain file with restrictive permissions, like `huggingface-cli login`'s own token file; the
  machine is the user's (Principle I). Not Hugging Face's own token file: in the container it isn't
  on a persistent volume.

## Out of Scope

- Creating the Hugging Face account or accepting licences on the user's behalf.
- Encrypting the token at rest.
- Other secrets.
