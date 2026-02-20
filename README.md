# Lattice

Lattice is an alternative, locally running open-source frontend for Weights & Biases (W&B).

It is intentionally minimal: load runs, browse groups, and plot metrics without using the hosted W&B UI.

## Run

```bash
uv sync
uv run python app.py
```

Open `http://127.0.0.1:8000`.

## Run With Docker

```bash
docker build -t lattice .
docker run --rm -p 8000:8000 -e FLASK_SECRET_KEY=change-me lattice
```

## Screenshots

Light mode
![Light mode](screenshots/light-mode.png)

Dark mode
![Dark mode](screenshots/dark-mode.png)
