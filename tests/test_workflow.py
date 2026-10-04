import contextlib
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import foliolense_crawl as app


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(app.SCHEMA)
        self.addCleanup(self.conn.close)

    def add_page(self, path):
        self.conn.execute(
            "INSERT INTO pages(path,page,file_sha256,dim,embedding,indexed_at) VALUES (?,?,?,?,?,?)",
            (str(path), 2, "hash", 2, np.array([3, 4], dtype=np.float16).tobytes(), 1),
        )
        self.conn.execute(
            "INSERT INTO files(path,size,mtime_ns,sha256,indexed_at) VALUES (?,0,0,'hash',1)",
            (str(path),),
        )
        self.conn.commit()

    def test_cleanup_is_scoped_and_preserves_existing_files(self):
        folder = self.root / "documents"
        folder.mkdir()
        live = folder / "live.pdf"
        live.touch()
        missing = folder / "deleted.pdf"
        outside = self.root / "elsewhere.pdf"
        for path in (live, missing, outside):
            self.add_page(path)
        self.assertEqual(app.prune_missing(self.conn, folder), 1)
        self.assertEqual({r[0] for r in self.conn.execute("SELECT path FROM pages")},
                         {str(live), str(outside)})
        self.assertIsNone(self.conn.execute("SELECT path FROM files WHERE path=?", (str(missing),)).fetchone())

    def test_permission_error_does_not_remove_document(self):
        path = self.root / "private.pdf"
        self.add_page(path)
        with patch.object(Path, "stat", side_effect=PermissionError):
            self.assertEqual(app.prune_missing(self.conn, self.root), 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0], 1)

    def test_missing_results_hidden_without_deleting_index(self):
        live = self.root / "live.pdf"
        live.touch()
        for path in (live, self.root / "missing.pdf"):
            self.add_page(path)
        matrix, meta, dim = app.load_matrix(self.conn, available_only=True)
        self.assertEqual(len(meta), 1)
        self.assertEqual(meta[0][1], str(live))
        np.testing.assert_allclose(matrix, [[0.6, 0.8]])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0], 2)

    def test_empty_folder_cleanup_needs_no_model(self):
        self.add_page(self.root / "deleted.pdf")
        args = SimpleNamespace(root=str(self.root), db=str(self.root / "index.sqlite"),
                               model=app.MODEL_ID, dim=512, dpi=144)
        with patch.object(app, "connect_db", return_value=self.conn), patch.object(app, "load_model") as model:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                app.crawl(args)
            model.assert_not_called()
            self.assertIn("No supported PDFs", output.getvalue())
        # crawl closes its connection; avoid closing it twice in cleanup (SQLite permits it).

    def test_opener_receives_path_as_one_argument(self):
        path = self.root / "a file;with spaces.pdf"
        path.touch()
        with patch.object(app.shutil, "which", return_value="/usr/bin/xdg-open"), patch.object(app.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                app.open_result(str(path), 2)
            self.assertEqual(run.call_args.args[0], ["/usr/bin/xdg-open", str(path)])
            self.assertIn("Go to page 2", output.getvalue())

    def test_opener_failure_is_readable(self):
        path = self.root / "file.pdf"
        path.touch()
        with patch.object(app.shutil, "which", return_value="xdg-open"), patch.object(app.subprocess, "run", side_effect=app.subprocess.TimeoutExpired("xdg-open", 10)):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                app.open_result(str(path), 1)
            self.assertIn("Open it manually", output.getvalue())

    def test_result_chooser_retries_and_returns(self):
        with patch("builtins.input", side_effect=["bad", "0", "2", "1", ""]), patch.object(app, "open_result") as opener:
            with contextlib.redirect_stdout(io.StringIO()):
                app.choose_result([(1, "/example.pdf", 2)])
            opener.assert_called_once_with("/example.pdf", 2)

    def test_chooser_handles_closed_input(self):
        with patch("builtins.input", side_effect=EOFError):
            with contextlib.redirect_stdout(io.StringIO()):
                app.choose_result([(1, "/example.pdf", 2)])

    def test_top_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            app.build_parser().parse_args(["query", "example", "--top", "0"])


if __name__ == "__main__":
    unittest.main()
