"""FGAO RAG inside LangChain: LangChain retrievers / models in, an LCEL Runnable out.

    pip install "fgao[langchain]" langchain-openai     # or any LangChain chat model
    python langchain_lcel.py                          # falls back to the offline toy LLM
"""
import os

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

from fgao import FGAORAG
from toy import KB, WEB, ToyLLM

if os.environ.get("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    model = ToyLLM()

# any LangChain vector store or retriever works as a source
kb = InMemoryVectorStore.from_documents([Document(page_content=d["text"], metadata={"title": d["title"]}) for d in KB],
                                        DeterministicFakeEmbedding(size=64))
rag = FGAORAG.from_retrievers(kb=kb, web=WEB, llm=model)

chain = rag.as_runnable() | RunnableLambda(lambda r: r["answer"] or "No verified answer.")
print(chain.invoke({"question": "What is the capital of Japan?"}))
print(chain.batch(["What is the capital of Germany?", "What is the capital of Spain?"]))

# or give it to an agent as a tool
tool = rag.as_tool()
print(tool.invoke({"question": "What is the capital of Kenya?"}))
