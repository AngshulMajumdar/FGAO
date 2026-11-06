"""FGAO RAG with a Hugging Face model.

    pip install "fgao[huggingface]"
    export HF_TOKEN=hf_...
    python quickstart.py
"""
import os

from fgao import FGAORAG
from fgao.integrations.huggingface import HFInferenceLLM

from toy import KB, WEB

llm = HFInferenceLLM(os.environ.get("FGAO_MODEL", "Qwen/Qwen2.5-7B-Instruct"), temperature=0)
rag = FGAORAG.from_retrievers(kb=KB, web=WEB, llm=llm)  # lists of documents are indexed with BM25

result = rag.invoke("What is the capital of Japan?")
print(result.answer)        # None means FGAO abstained: no grounded, supported answer was found
print(result.explain())     # every decision with its numerical justification
