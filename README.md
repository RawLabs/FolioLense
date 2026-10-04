# FolioLense

Find relevant pages in your local PDFs and images by describing what you
want to find. FolioLense runs in a terminal and saves its progress so you
can stop and resume. Unchanged documents are skipped on later runs.

The search model is `tomaarsen/Qwen3-VL-Embedding-2B-vdr`.

## Requirements

- Linux with Bash and Python 3 with `venv` support.
- An NVIDIA GPU, working NVIDIA driver, and CUDA-capable PyTorch for the installer.
- Internet access for dependencies and the initial model download.
- Disk space for the model, environment, and document index.

## Install and launch

Open a terminal in this project folder, then run:

```bash
./install.sh
foliolense
```

The installer stores the application and its virtual environment in
`~/.local/share/foliolense/` and adds a launcher at `~/.local/bin/foliolense`.
Ensure `~/.local/bin` is on your PATH, or launch
`~/.local/share/foliolense/foliolense.sh` directly.

The first indexing or search run downloads the model. The menu supports
indexing a folder, searching, and viewing index statistics. The index is
stored at `~/.local/share/foliolense/index.sqlite`.

## Your first search

1. Choose **1) Add or update searchable documents**.
2. Enter a folder or file path, or press Enter to use the suggested folder.
   Paste the path without surrounding quotes; spaces in paths are supported.
3. Wait for document preparation to finish. The first run also downloads
   the search model and can take a while.
4. Choose **2) Search documents** and describe what you need, such as
   `shift schedules` or `a chart of monthly costs`.
5. Results show a relevance score, page number, and file path. Higher scores
   mean a closer match; they are not percentages. Open the file in your
   usual viewer and go to the listed page. Enter a result number to open
   its file, or press Enter to return to the menu.

Choose **3) Show document summary** to see progress and file errors, or
**4) Quit** to leave. Run option 1 again to add or update documents.

Supported formats: PDF, PNG, JPG/JPEG, WebP, BMP, and TIFF.
Ctrl+C preserves completed pages so a later crawl can resume.

## Command-line usage

Using the installed environment:

```bash
~/.local/share/foliolense/.venv/bin/python ~/.local/share/foliolense/foliolense_crawl.py crawl ~/Documents --db ~/.local/share/foliolense/index.sqlite
~/.local/share/foliolense/.venv/bin/python ~/.local/share/foliolense/foliolense_crawl.py query "shift schedules" --db ~/.local/share/foliolense/index.sqlite --top 15
~/.local/share/foliolense/.venv/bin/python ~/.local/share/foliolense/foliolense_crawl.py stats --db ~/.local/share/foliolense/index.sqlite
```

Direct CLI calls default to `foliolense-index.sqlite` in the current directory.
`crawl` and `query` accept `--device cpu`, although the installer requires
an NVIDIA GPU and CPU inference can be slow.

## Batch queries

`foliolense_batch.py` loads the model and page matrix once for multiple
queries. Create a JSONL file with one query object per line:

```json
{"query_id":"q1","query":"shift schedules"}
```

Optional fields: `cluster_id`, `cluster_label`, and `kind`.

```bash
~/.local/share/foliolense/.venv/bin/python ~/.local/share/foliolense/foliolense_batch.py --queries health_queries.jsonl --db ~/.local/share/foliolense/index.sqlite --output foliolense_results.jsonl --top 15
```

The batch script supports both the crawler's path-based schema and an
alternate schema with a `file_paths` table. Results contain scores, page
numbers, source hashes, and local file paths.

## Current limitations

The menu opens selected files using `xdg-open` and your default viewer.
Go to the listed page manually; automatic page navigation depends on the
viewer and is not supported yet. CLI users can add `--interactive` to
`query` to get the same result chooser.
Updating a folder removes index entries for files that are no longer
there. Other indexed folders are kept. Missing files are hidden from
search results even before you update their folder. The installer currently requires an NVIDIA GPU.

## Local data

Indexes, query/result JSONL files, environments, caches, and local
credentials are ignored by Git. Indexes and results can contain private
document paths and should remain local. Model weights are downloaded
separately and are not included in this repository.
