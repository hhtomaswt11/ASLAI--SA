# TrabalhoPraticoNovo

Novo prototipo final do projeto ASLAI.

## Estrutura

- `frontend/`: aplicacao Next.js com Tailwind para captura de webcam e UX.
- `backend/`: API FastAPI para inferencia, correcao de frase e texto para voz.
- `notebooks/`: exploracao, treino e validacao de artefactos.
- `shared_models/`: destino recomendado para os modelos do prototipo final.

## Estado atual

Implementacao atual inclui:

- backend FastAPI modular com inferencia estatica e dinamica (MediaPipe + MLP/LSTM);
- frontend Next.js compilavel e ligado aos endpoints principais;
- politica de dependencias Python com MediaPipe recente;
- notebooks reorganizados para overview, pipelines estatico/dinamico e validacao;

## Notebooks

- `00_project_overview.ipynb`
- `01_model_validation.ipynb`
- `02_static_pipeline.ipynb`
- `03_dynamic_pipeline.ipynb`
- `04_inference_validation.ipynb`

## Arranque local

### Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Por omissao, o frontend consome `http://localhost:8000/api`.
