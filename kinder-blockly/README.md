# kinder-blockly

Visual programming interface for KinDER robot skills using Google Blockly.

## Installation

### Backend (Python)

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[develop]"
```

### Frontend (Svelte)

```bash
cd frontend
npm install
```

## Development

Start the backend server:

```bash
python -m kinder_blockly
```

In a separate terminal, start the Svelte dev server with hot reload:

```bash
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## Production build

Build the frontend and serve via Flask:

```bash
cd frontend
npm run build
```

Then run the backend as normal — Flask serves the compiled frontend from `src/kinder_blockly/static/`:

```bash
python -m kinder_blockly
```

Then open http://127.0.0.1:5000 in your browser.
