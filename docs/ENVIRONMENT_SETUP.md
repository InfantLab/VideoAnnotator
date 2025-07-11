# Environment Setup Guide

## Setting up HuggingFace Authentication

The VideoAnnotator project uses PyAnnote for speaker diarization, which requires a HuggingFace token.

### 1. Get a HuggingFace Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

### 2. Configure Your Environment

#### Option A: Using .env file (Recommended)

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your token:
   ```bash
   # HuggingFace Authentication Token
   HF_AUTH_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

#### Option B: Using System Environment Variables

Set the environment variable in your shell:

**Windows (PowerShell):**
```powershell
$env:HF_AUTH_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Windows (Command Prompt):**
```cmd
set HF_AUTH_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Linux/macOS:**
```bash
export HF_AUTH_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
- Verify the token is set as `HF_AUTH_TOKEN=your_token_here`
- Ensure no extra spaces around the equals sign

**"Authentication failed"**
- Verify your token is valid at [HuggingFace Settings](https://huggingface.co/settings/tokens)
- Make sure you've accepted the terms for pyannote/speaker-diarization-3.1
- Try regenerating your token if it's old

**"Model access denied"**
- Go to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- Click "Agree and access repository"
- Wait a few minutes for permissions to propagate
