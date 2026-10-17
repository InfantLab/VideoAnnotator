# Environment Setup Guide

## Setting up HuggingFace Authentication

The VideoAnnotator project uses PyAnnote for speaker diarization, which requires a HuggingFace token.

### 1. Get a HuggingFace Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Signed in as the same account, accept the licence on each model page the viewer lists on the
   Speaker turns card (today `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0` and
   `pyannote/speaker-diarization-community-1`)

### 2. Give the token to VideoAnnotator

No file to edit and no restart, whichever way you choose. The token stays on the server's machine.

#### Option A: In the viewer (recommended)

On the **Pipelines** page (or when choosing pipelines for a run), the Speaker turns card shows the
three steps above. Paste the token into step 3 and press **Save and check**: the card names the
token's account and shows each licence as accepted or not. You need an admin key (the first key a
new install creates is one). To see, change or remove the token later: **Settings → API
Configuration → Hugging Face token**.

#### Option B: From a terminal

```bash
videoannotator settings huggingface-token            # asks for it, without showing it
videoannotator settings huggingface-token --status   # set? which account?
videoannotator settings huggingface-token --remove
```

With no terminal (a script or an agent), it reads the token from standard input:
`printf '%s\n' "$TOKEN" | videoannotator settings huggingface-token`.

Options A and B save the token in a file next to the database (`huggingface_token`, readable only
by you), so in the container it survives being recreated.

#### Option C: An environment variable

A token in the server's environment takes priority over a saved one, and changing it needs a
restart. Use this for setups that already manage their environment.

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your token:
   ```bash
   # HuggingFace Authentication Token
   HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

Or set it in your shell:

**Windows (PowerShell):**

```powershell
$env:HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Windows (Command Prompt):**

```cmd
set HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Linux/macOS:**

```bash
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Test Your Setup

Run the diarization test script:

```bash
python test_diarization.py
```

If configured correctly, you should see:

```
=== Testing PyAnnote Diarization Pipeline ===

1. Checking Prerequisites:
   FFmpeg available: True
   HuggingFace token: ✓ Found
```

### 4. Security Notes

- **Never commit your `.env` file** - it's already in `.gitignore`
- Use the `.env.example` file as a template for team members
- Keep your HuggingFace token secure and don't share it publicly
- Rotate your token periodically for security

### 5. Troubleshooting

**"HuggingFace token: ✗ Not found"**

- Check that your `.env` file exists in the project root
- Verify the token is set as `HUGGINGFACE_TOKEN=your_token_here`
- Older setups using `HF_AUTH_TOKEN` (or Hugging Face's own `HF_TOKEN`) still work; `HUGGINGFACE_TOKEN` wins if both are set
- Ensure no extra spaces around the equals sign

**"Authentication failed"**

- Verify your token is valid at [HuggingFace Settings](https://huggingface.co/settings/tokens)
- Make sure you've accepted the terms for pyannote/speaker-diarization-3.1
- Try regenerating your token if it's old

**"Model access denied"**

- Go to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- Click "Agree and access repository"
- Wait a few minutes for permissions to propagate
