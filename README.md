# Privacy Website Viewer

This project crawls a privacy or policy webpage, follows links in the main content area, flags dead links, and exports the results as graph and CSV files.

## Requirements

- Python 3.14 or newer
- `uv` recommended for dependency management

## Install

### Option 1: `uv` (recommended)

Install dependencies from the project lockfile:

```bash
uv sync
```

Run the script with:

```bash
uv run python main.py <URL>
```

Example:

```bash
uv run python main.py https://www.gatech.edu/privacy
```

### Option 2: standard virtual environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -e .
```

Run the script:

```bash
python main.py <URL>
```

## Usage

Basic command:

```bash
python main.py <URL>
```

Useful options:

- `--output-dir output` to choose where files are written
- `--max-depth 4` to limit recursive crawl depth
- `--max-pages 200` to cap the number of crawled pages
- `--timeout 20` to control request timeout
- `--delay 0.25` to slow requests between pages
- `--allowed-domain example.com` to restrict recursive crawling to specific domains
- `--follow-external` to recursively crawl external links too
- `--insecure` to disable TLS certificate verification

Example with options:

```bash
uv run python main.py https://www.gatech.edu/privacy \
  --max-depth 2 \
  --max-pages 50 \
  --output-dir output
```

## Output

The crawler writes files to the `output/` directory by default, including:

- `*.graphml` for graph tools
- `*.networkx.pkl` for Python/NetworkX reuse
- `*.networkx.png` for a static graph image when `matplotlib` is available
- `*.html` for an interactive graph when `pyvis` is available
- `*.nodes.csv` and `*.edges.csv` for tabular inspection
- `*.summary.json` for crawl metadata

## Notes

- If you run `python main.py ...` before installing dependencies, the script will fail with missing-module errors such as `ModuleNotFoundError: No module named 'networkx'`.
- If `uv` is not installed, see: https://docs.astral.sh/uv/
