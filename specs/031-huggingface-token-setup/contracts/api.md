# Contract: `/api/v1/system/huggingface-token`

`TokenStatus` (all three return it):

```json
{"set": true, "source": "saved", "env_var": null, "account": "caspar", "check": "ok"}
```

- `source`: `"environment"` | `"saved"` | `null`; `env_var`: the variable that set it when
  `source` is `"environment"`; `check`: `"ok"` | `"invalid"` | `"unknown"` | `null` (not set);
  `account`: from Hugging Face, when known. Never the token.

| Method | Who | Body | Result |
|---|---|---|---|
| GET | any signed-in user | — | 200 `TokenStatus` |
| PUT | admin | `{"token": "hf_…"}` | 200 `TokenStatus` (saved; `check` ok or unknown); 400 `HF_TOKEN_REJECTED` (not saved); 400 `HF_TOKEN_EMPTY` |
| DELETE | admin | — | 200 `TokenStatus` (after removal; still set if the environment sets one) |

Non-admin PUT/DELETE: 403 as other admin routes.
