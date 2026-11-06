import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples")
OFFLINE = {"offline_demo.py": None, "workflow.py": None, "controller_step_api.py": None, "async_batch.py": None,
           "langchain_lcel.py": "langchain_core", "langgraph_node.py": "langgraph", "agents.py": None,
           "mcp_tools.py": "mcp"}


@pytest.mark.parametrize("name", sorted(OFFLINE))
def test_example_runs(name):
    dep = OFFLINE[name]
    if dep and importlib.util.find_spec(dep) is None:
        pytest.skip(f"{dep} not installed")
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    import fgao

    env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(os.path.dirname(fgao.__file__)), env.get("PYTHONPATH", "")])
    out = subprocess.run([sys.executable, name], cwd=EX, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip()


def test_quickstart_compiles():
    import py_compile

    py_compile.compile(os.path.join(EX, "quickstart.py"), doraise=True)


def test_space_app():
    pytest.importorskip("gradio")
    sys.path.insert(0, os.path.join(ROOT, "space"))
    sys.path.insert(0, EX)
    try:
        import app
        from toy import ToyLLM
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    head, rows, d = app.answer("What is the capital of Japan?", app.SAMPLE_KB, app.SAMPLE_WEB, "", "", 16, llm=ToyLLM())
    assert "Tokyo" in head and d["answer"] == "Tokyo" and rows[-1][1] == "verify()"
    head, _, _ = app.answer("What is the capital of Spain?", app.SAMPLE_KB, "", "", "", 16, llm=ToyLLM())
    assert "Abstained" in head
    assert "Enter a question" in app.answer("", app.SAMPLE_KB, "", "", "", 16)[0]
    os.environ.pop("HF_TOKEN", None)
    assert "HF_TOKEN" in app.answer("q?", app.SAMPLE_KB, "", "", "", 16)[0]
    assert app.demo is not None


@pytest.mark.parametrize("name", ["make_corpus.py", "fake_openai.py", "loadgen.py"])
def test_loadtest_tools_compile(name):
    import py_compile

    py_compile.compile(os.path.join(ROOT, "loadtest", name), doraise=True)


def test_make_corpus_runs(tmp_path):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "loadtest", "make_corpus.py"), "--entities", "50",
                          "--filler", "100", "--questions", "40", "--out", str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert {p.name for p in tmp_path.iterdir()} == {"kb.jsonl", "web.jsonl", "questions.jsonl"}
