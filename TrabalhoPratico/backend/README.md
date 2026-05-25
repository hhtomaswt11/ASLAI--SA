# Backend

FastAPI service para inferencia ASL.

## Ambiente recomendado

- Python 3.11
- pip atualizado

## Instalar

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Executar

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /api/health`
- `POST /api/predict`
- `POST /api/predict_dynamic`
- `POST /api/llm_correct`
- `POST /api/speak`

## Modelos

Por omissao, o backend tenta carregar artefactos de:

`TrabalhoPraticoNovo/shared_models`

## Fluxo recomendado de validacao

No diretorio raiz `TrabalhoPraticoNovo`:

```bash
python scripts/copy_legacy_models.py
python scripts/validate_model_artifacts.py
python scripts/smoke_test_api.py
```
