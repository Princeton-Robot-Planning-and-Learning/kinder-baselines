# Lab setup — do this **before** the lab

This installs only what the lab needs (a few minutes). `uv` downloads the right
Python for you, so nothing else needs to be preinstalled. If step 3 fails, message
the instructors **before** the session.

## 1. Install `uv`

The package manager — see https://docs.astral.sh/uv/getting-started/installation/, or:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then **restart your terminal** so `uv` is on your `PATH`.

## 2. Clone the repository

```bash
git clone https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines.git
cd kinder-baselines
```

## 3. Create the environment and install the lab packages

Run this from the `kinder-baselines` root. **Activate the environment before
installing** — that way the packages land in this venv even if you already have
another environment active (e.g. conda's `base`):

```bash
uv venv --python=3.10
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e "./kinder-models[develop]" -e "./kinder-bilevel-planning[develop]"
```

(This installs just the two packages the lab uses and their dependencies — not the
other baselines.)

### Slim install (no PyBullet — smaller and faster)

The command above pulls every simulation backend, including PyBullet and MuJoCo.
The lab only uses the 2D (Obstruction2D) environments, which need neither. To
install just those, install the packages without their dependencies and then the
lab's 2D requirements file:

```bash
uv venv --python=3.10
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install --no-deps kindergarden \
    -e ./kinder-models -e ./kinder-bilevel-planning
uv pip install -r kinder-bilevel-planning/lab/requirements/lab2d.txt
```

This produces the same working lab with a much lighter install — useful on
machines (or hosted notebooks like Colab) where the full backend stack is slow or
unnecessary.

## 4. Go to the lab and verify

```bash
cd kinder-bilevel-planning/lab
python -c "import kinder, kinder_models, bilevel_planning, kinder_bilevel_planning; print('install OK')"
python -m pytest part1_stacking -q
```

You're ready if:

- you see `install OK`, and
- the tests **fail** with messages mentioning `TODO(1)` / `TODO(2)` (that's the
  unfinished exercise — *not* an error). If you instead see `ModuleNotFoundError`,
  the install didn't finish.

Then open `README.md` and start with **Part 1**.

## If something breaks

- **`uv: command not found`** after step 1 → restart your terminal (or open a new
  one) so the install lands on your `PATH`.
- **Step 3 errors about a missing file / package** → make sure you're in the
  `kinder-baselines` root directory (the one containing `kinder-models/` and
  `kinder-bilevel-planning/`).
- **`ModuleNotFoundError` in step 4** → your prompt should show `(.venv)`; if not,
  run `source .venv/bin/activate` and re-run the `uv pip install` from step 3 (if
  you installed with another env active, the packages may have gone there instead).
- Still stuck? Send us the exact command you ran and the full error output.
