#!/usr/bin/env python3
"""Unit tests for kb_retrieve.py. Run with:

    PYTHONPATH=/home/jamjee/workplace/aiworkspace/.pytools \
      /apollo/env/envImprovement/bin/python3.12 -m unittest \
      .github/scripts/test_kb_retrieve.py

Uses a fake bedrock-agent-runtime client -- no AWS calls.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kb_retrieve  # noqa: E402


SAMPLE_DIFF = """\
diff --git a/sagemaker-core/src/sagemaker_core/model.py b/sagemaker-core/src/sagemaker_core/model.py
index 111..222 100644
--- a/sagemaker-core/src/sagemaker_core/model.py
+++ b/sagemaker-core/src/sagemaker_core/model.py
@@ -1,3 +1,8 @@
+def build_model(config):
+    return config
+
+class ModelBuilder:
+    async def deploy(self):
+        pass
-def old_helper():
 unchanged line
diff --git a/sagemaker-train/src/train.py b/sagemaker-train/src/train.py
index 333..444 100644
--- a/sagemaker-train/src/train.py
+++ b/sagemaker-train/src/train.py
@@ -1 +1,2 @@
+    def _internal(self):
"""


def _result(uri, title=None, source_url=None, score=0.5, text="body"):
    meta = {}
    if title:
        meta["title"] = title
    if source_url:
        meta["source_url"] = source_url
    return {
        "content": {"text": text},
        "location": {"s3Location": {"uri": uri}},
        "metadata": meta,
        "score": score,
    }


class FakeClient:
    """Returns a canned response per query text; records queries seen."""

    def __init__(self, per_query):
        self._per_query = per_query
        self.queries = []

    def retrieve(self, knowledgeBaseId, retrievalQuery, retrievalConfiguration):
        q = retrievalQuery["text"]
        self.queries.append(q)
        return {"retrievalResults": self._per_query.get(q, [])}


class ExtractQueriesTest(unittest.TestCase):
    def test_extracts_title_paths_and_names(self):
        queries = kb_retrieve.extract_queries(SAMPLE_DIFF, "Fix model builder deploy")
        self.assertEqual(queries[0], "Fix model builder deploy")
        self.assertIn("sagemaker-core/src/sagemaker_core/model.py", queries)
        self.assertIn("sagemaker-train/src/train.py", queries)
        # def / class / async def names from added lines
        self.assertIn("build_model", queries)
        self.assertIn("ModelBuilder", queries)
        self.assertIn("deploy", queries)
        self.assertIn("_internal", queries)
        # removed lines (old_helper) and the +++ header path must not leak in
        self.assertNotIn("old_helper", queries)

    def test_respects_max_queries_cap(self):
        big_title = "t"
        diff_lines = ["diff --git a/f b/f", "--- a/f", "+++ b/f"]
        for i in range(50):
            diff_lines.append("+def func_{}():".format(i))
        queries = kb_retrieve.extract_queries("\n".join(diff_lines), big_title)
        self.assertLessEqual(len(queries), kb_retrieve.MAX_QUERIES)

    def test_empty_diff_no_title(self):
        self.assertEqual(kb_retrieve.extract_queries("", ""), [])


class RetrieveDedupeTest(unittest.TestCase):
    def test_dedupes_by_source_url_keeping_best_score(self):
        dup_low = _result("s3://b/doc1.md", source_url="https://x/doc1", score=0.30)
        dup_high = _result("s3://b/doc1.md", source_url="https://x/doc1", score=0.90)
        other = _result("s3://b/doc2.md", source_url="https://x/doc2", score=0.40)
        client = FakeClient({"q1": [dup_low], "q2": [dup_high, other]})
        hits = kb_retrieve.retrieve(client, "KBID", ["q1", "q2"])
        urls = [h["url"] for h in hits]
        self.assertEqual(urls.count("https://x/doc1"), 1)
        self.assertEqual(len(hits), 2)
        doc1 = next(h for h in hits if h["url"] == "https://x/doc1")
        self.assertEqual(doc1["score"], 0.90)
        # sorted by score descending
        self.assertEqual(hits[0]["url"], "https://x/doc1")

    def test_dedupes_by_location_when_no_source_url(self):
        a = _result("s3://b/same.md", score=0.5)
        b = _result("s3://b/same.md", score=0.6)
        client = FakeClient({"q": [a, b]})
        hits = kb_retrieve.retrieve(client, "KBID", ["q"])
        self.assertEqual(len(hits), 1)


class RenderCapTest(unittest.TestCase):
    def test_render_caps_output(self):
        hits = [
            {"score": 0.9, "text": "x" * 5000, "title": "T1", "url": "https://x/1"},
            {"score": 0.8, "text": "y" * 5000, "title": "T2", "url": "https://x/2"},
            {"score": 0.7, "text": "z" * 5000, "title": "T3", "url": "https://x/3"},
        ]
        out = kb_retrieve.render(hits, max_chars=6000)
        self.assertLessEqual(len(out), 6000 + 200)
        self.assertIn("omitted to stay within the size cap", out)
        self.assertIn("T1", out)
        self.assertNotIn("T3", out)

    def test_render_empty(self):
        out = kb_retrieve.render([], max_chars=12000)
        self.assertIn("No relevant team precedent", out)


class FailurePathTest(unittest.TestCase):
    def test_main_writes_file_and_exits_zero_on_failure(self):
        # No --kb-id => ValueError inside main => must still write out + exit 0.
        with tempfile.TemporaryDirectory() as d:
            diff_path = os.path.join(d, "pr.diff")
            out_path = os.path.join(d, "kb.md")
            with open(diff_path, "w") as fh:
                fh.write(SAMPLE_DIFF)
            rc = kb_retrieve.main(
                ["--diff", diff_path, "--title", "t", "--out", out_path]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path) as fh:
                content = fh.read()
            self.assertIn("Knowledge base context", content)

    def test_main_writes_file_when_diff_missing(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "kb.md")
            rc = kb_retrieve.main(
                ["--diff", os.path.join(d, "nope.diff"),
                 "--kb-id", "KBID", "--out", out_path]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
