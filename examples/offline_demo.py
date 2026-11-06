"""FGAO RAG end to end without network access (rule-based stand-in LLM).

    python offline_demo.py
"""
from fgao import FGAORAG

from toy import KB, WEB, ToyLLM

rag = FGAORAG.from_retrievers(kb=KB, web=WEB, llm=ToyLLM())

for q in ["What is the capital of France?",   # in the knowledge base
          "What is the capital of Japan?",    # only on the web source: FGAO escalates
          "What is the capital of Spain?"]:   # nowhere: FGAO abstains
    r = rag.invoke(q)
    print(r.explain())
    print()
