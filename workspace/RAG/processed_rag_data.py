"""RAG 语料构建脚本：确定性的 CLI 入口。

导入本模块不再触发构建（原脚本在模块顶层直接建库）；只有作为
__main__ 执行并显式给出参数时才构建。构建后写出 index manifest
（schema_version、语料清单与摘要、embedding 摘要、分块参数、collection
文档数与文档 ID），供试用的检索证据链使用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def load_documents(directory: str, chunk_size: int, chunk_overlap: int, separators: list) -> list:
    """加载目录中的 txt 语料并按参数分块。"""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    loader = DirectoryLoader(
        path=directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    if not documents:
        raise SystemExit(f"语料目录为空或不含 txt 文件: {directory}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=separators,
    )
    split_docs = text_splitter.split_documents(documents)
    if not split_docs:
        raise SystemExit("分块结果为空，请检查语料内容与分隔符")
    return split_docs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_manifest(corpus_dir: Path, embedding_model, split_docs: list, args) -> dict:
    corpus_files = sorted(p for p in corpus_dir.rglob("*.txt") if p.is_file())
    return {
        "schema_version": 1,
        "corpus": {
            "directory": corpus_dir.as_posix(),
            "files": [
                {"path": p.relative_to(corpus_dir.parent).as_posix(), "sha256": file_sha256(p)}
                for p in corpus_files
            ],
        },
        "embedding_model": getattr(embedding_model, "model_path", "<configured-embedding>"),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "separators": list(args.separator),
        "persist_directory": args.persist_directory,
        "chunk_count": len(split_docs),
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="从固定语料构建 Chroma 索引并写出构建清单")
    parser.add_argument("--corpus", required=True, help="语料目录（含 .txt 文件）")
    parser.add_argument("--embedding-model", required=True, help="GTE 本地模型目录")
    parser.add_argument("--persist-directory", required=True, help="Chroma 持久化输出目录")
    parser.add_argument("--manifest", required=True, help="构建清单输出 JSON 路径")
    parser.add_argument("--chunk-size", type=int, required=True, help="分块大小（字符）")
    parser.add_argument("--chunk-overlap", type=int, required=True, help="分块重叠（字符）")
    parser.add_argument(
        "--separator",
        action="append",
        required=True,
        help="递归切分分隔符，可多次给出（按给定顺序使用）",
    )
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        raise SystemExit(f"语料目录不存在: {corpus_dir}")
    persist_dir = Path(args.persist_directory)
    if persist_dir.exists() and any(persist_dir.iterdir()):
        raise SystemExit(f"持久化目录非空，拒绝覆盖: {persist_dir}")
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        raise SystemExit(f"构建清单已存在，拒绝覆盖: {manifest_path}")

    from gteEmbeddings import GTEEmbeddings

    embeddings = GTEEmbeddings(model_path=args.embedding_model)
    split_docs = load_documents(
        str(corpus_dir), args.chunk_size, args.chunk_overlap, list(args.separator)
    )

    persist_dir.mkdir(parents=True, exist_ok=True)
    db = Chroma.from_documents(
        documents=split_docs, embedding=embeddings, persist_directory=str(persist_dir)
    )
    db.persist()

    manifest = build_manifest(corpus_dir, embeddings, split_docs, args)
    manifest["collection_count"] = db._collection.count()
    manifest["doc_ids"] = [str(item) for item in db._collection.get()["ids"]]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"索引构建完毕: {persist_dir}（{manifest['chunk_count']} 块 / 库内 {manifest['collection_count']} 条）")
    print(f"构建清单: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
