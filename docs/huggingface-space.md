# Hugging Face Spaces

The repository holds two Spaces. Neither contains benchmark results.

| folder | Space type | what it serves |
|---|---|---|
| `space/` | Gradio | a playground (passages + question → answer and decision trace) and the API reference |
| `space-api/` | Docker | the HTTP API (`fgao serve` on port 7860): `/v1/answer`, `/v1/answer/stream`, `/v1/batch`, `/docs` |

FGAO itself needs no GPU: both Spaces run on the free CPU hardware. The LLM is called through Hugging Face Inference Providers, so both need an `HF_TOKEN` secret (Settings → Variables and secrets).

## Gradio playground (`space/`)

1. Create a Space with the **Gradio** SDK.
2. Upload `space/app.py`, `space/README.md`, `space/requirements.txt`, the package folder `src/fgao/` (as `fgao/` next to `app.py`), and `docs/` (the Space README links to it).
3. Add the `HF_TOKEN` secret, or let each visitor paste a token in the UI. To change the default model, set `FGAO_MODEL`.

Run it locally:

```bash
pip install -e ".[demo]"
python space/app.py
```

## HTTP API (`space-api/`)

1. Create a Space with the **Docker** SDK.
2. Upload the contents of `space-api/` (`README.md`, `Dockerfile`, `fgao.yaml`, `data/`), plus `pyproject.toml` and `src/fgao/` at the Space root, keeping `src/fgao/`.
3. Add the `HF_TOKEN` secret. To require API keys, also add `FGAO_API_KEYS=key1,key2`.

`fgao.yaml` sets the model (`Qwen/Qwen2.5-7B-Instruct` through `https://router.huggingface.co/v1`), the limits and the default corpus (`data/kb.jsonl`). Clients can also send their own passages with each request.

Test the image locally before uploading:

```bash
mkdir /tmp/fgao-api && cp -r space-api/* pyproject.toml /tmp/fgao-api/ && mkdir /tmp/fgao-api/src && cp -r src/fgao /tmp/fgao-api/src/
docker build -t fgao-api /tmp/fgao-api
docker run -p 7860:7860 -e HF_TOKEN=hf_... fgao-api
```

## With the Hugging Face CLI

```bash
pip install huggingface_hub && hf auth login
hf repos create <user>/FGAO-API --repo-type space --space-sdk docker
hf upload <user>/FGAO-API /tmp/fgao-api . --repo-type=space
```
