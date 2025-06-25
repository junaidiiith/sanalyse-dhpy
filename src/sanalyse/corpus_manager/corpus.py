# tuwanalyse/corpus.py

import os
import glob
import re
import pickle
import random
from typing import List, Dict, Tuple, Callable, Optional, Any

import pandas as pd
from langdetect import detect
from lxml import etree
from concurrent.futures import ThreadPoolExecutor

# Optional imports for advanced features
try:
    from whoosh import index
    from whoosh.fields import Schema, TEXT, ID
    WHOOSH_AVAILABLE = True
except ImportError:
    WHOOSH_AVAILABLE = False

# Simple Document container
class Document:
    def __init__(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        self.id = doc_id
        self.text = text
        self.metadata = metadata

class Corpus:
    def __init__(self, documents: Optional[Dict[str, Document]] = None):
        """
        documents: mapping from doc_id → Document
        """
        self.documents: Dict[str, Document] = documents or {}

    # --------------------
    # Initialization / Loading
    # --------------------
    @classmethod
    def from_directory(cls, path: str, pattern: str = "*.txt", language: Optional[str] = None):
        docs = {}
        for filepath in glob.glob(os.path.join(path, pattern)):
            doc_id = os.path.splitext(os.path.basename(filepath))[0]
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            meta = {}
            if language is None:
                try:
                    meta['language'] = detect(text)
                except Exception:
                    meta['language'] = None
            else:
                meta['language'] = language
            docs[doc_id] = Document(doc_id, text, meta)
        return cls(docs)

    @classmethod
    def from_csv(cls, path: str, text_col: str, meta_cols: List[str] = [], language: Optional[str] = None):
        df = pd.read_csv(path, dtype=str).fillna('')
        docs = {}
        for idx, row in df.iterrows():
            doc_id = str(idx)
            text = row[text_col]
            meta = {col: row[col] for col in meta_cols}
            if language:
                meta['language'] = language
            else:
                try:
                    meta['language'] = detect(text)
                except Exception:
                    meta['language'] = None
            docs[doc_id] = Document(doc_id, text, meta)
        return cls(docs)

    @classmethod
    def from_tei(cls, path: str, xpath: str = "//TEI"):
        parser = etree.XMLParser(recover=True)
        tree = etree.parse(path, parser)
        elements = tree.xpath(xpath, namespaces=tree.getroot().nsmap)
        docs = {}
        for i, el in enumerate(elements):
            text = "".join(el.xpath(".//text()"))
            doc_id = el.get("xml:id") or f"tei_{i}"
            docs[doc_id] = Document(doc_id, text, {'source': path})
        return cls(docs)

    @classmethod
    def from_api(cls, source: str, **credentials):
        # Placeholder for any remote API loader
        raise NotImplementedError("API loaders must be implemented per source.")

    # --------------------
    # Accessors
    # --------------------
    def get_document_ids(self) -> List[str]:
        return list(self.documents.keys())

    def get_text(self, doc_id: str) -> str:
        return self.documents[doc_id].text

    def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        return dict(self.documents[doc_id].metadata)

    # --------------------
    # Language Handling
    # --------------------
    def detect_language(self, doc_id: str) -> str:
        text = self.get_text(doc_id)
        try:
            return detect(text)
        except Exception:
            return 'unknown'

    def force_language(self, doc_id: str, lang: str):
        self.documents[doc_id].metadata['language'] = lang

    # --------------------
    # Previews & Sampling
    # --------------------
    def head(self, n: int = 5) -> 'Corpus':
        ids = self.get_document_ids()[:n]
        return Corpus({i: self.documents[i] for i in ids})

    def tail(self, n: int = 5) -> 'Corpus':
        ids = self.get_document_ids()[-n:]
        return Corpus({i: self.documents[i] for i in ids})

    def random_sample(self, n: int = 10, seed: Optional[int] = None) -> 'Corpus':
        ids = self.get_document_ids()
        rnd = random.Random(seed)
        chosen = rnd.sample(ids, min(n, len(ids)))
        return Corpus({i: self.documents[i] for i in chosen})

    # --------------------
    # Filtering & Subsetting
    # --------------------
    def filter_by_metadata(self, **criteria) -> 'Corpus':
        def match(meta):
            return all(meta.get(k) == v for k, v in criteria.items())
        filtered = {
            did: doc for did, doc in self.documents.items()
            if match(doc.metadata)
        }
        return Corpus(filtered)

    def filter_by_text(self, regex: str) -> 'Corpus':
        pat = re.compile(regex)
        filtered = {
            did: doc for did, doc in self.documents.items()
            if pat.search(doc.text)
        }
        return Corpus(filtered)

    # --------------------
    # Merging & Splitting
    # --------------------
    def merge(self, other: 'Corpus') -> 'Corpus':
        merged = dict(self.documents)
        merged.update(other.documents)
        return Corpus(merged)

    def split(self, ratio: float = 0.8, seed: Optional[int] = None) -> Tuple['Corpus','Corpus']:
        ids = self.get_document_ids()
        rnd = random.Random(seed)
        rnd.shuffle(ids)
        split_at = int(len(ids) * ratio)
        train_ids, test_ids = ids[:split_at], ids[split_at:]
        return (
            Corpus({i: self.documents[i] for i in train_ids}),
            Corpus({i: self.documents[i] for i in test_ids})
        )

    # --------------------
    # Basic Statistics
    # --------------------
    @property
    def num_documents(self) -> int:
        return len(self.documents)

    @property
    def vocabulary_size(self) -> int:
        vocab = set()
        for doc in self.documents.values():
            vocab.update(doc.text.split())
        return len(vocab)

    def doc_length_distribution(self) -> pd.DataFrame:
        data = [(doc.id, len(doc.text.split())) for doc in self.documents.values()]
        df = pd.DataFrame(data, columns=['doc_id','num_tokens'])
        return df.describe()

    # --------------------
    # Export & Serialization
    # --------------------
    def to_folder(self, out_dir: str, fmt: str = "txt") -> None:
        os.makedirs(out_dir, exist_ok=True)
        for doc in self.documents.values():
            path = os.path.join(out_dir, f"{doc.id}.{fmt}")
            with open(path, 'w', encoding='utf-8') as f:
                if fmt == 'txt':
                    f.write(doc.text)
                elif fmt == 'json':
                    f.write(pd.Series({'id':doc.id, 'text':doc.text, **doc.metadata}).to_json())
                elif fmt == 'csv':
                    df = pd.DataFrame([{'id':doc.id, 'text':doc.text, **doc.metadata}])
                    df.to_csv(f, index=False)
                else:
                    raise ValueError(f"Unsupported format: {fmt}")

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for doc in self.documents.values():
            row = {'id': doc.id, 'text': doc.text}
            row.update(doc.metadata)
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls, path: str) -> 'Corpus':
        with open(path, 'rb') as f:
            docs = pickle.load(f)
        return cls(docs)

    # --------------------
    # Normalization Helpers
    # --------------------
    @staticmethod
    def unicode_normalize(text: str) -> str:
        return text.normalize('NFC')

    @staticmethod
    def collapse_whitespace(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def lowercase(text: str) -> str:
        return text.lower()

    # --------------------
    # Advanced Features (stubs / minimal)
    # --------------------
    def lazy_load(self):
        raise NotImplementedError("Lazy loading not yet implemented.")

    def parallel_apply(self, func: Callable[[Document], Any], workers: int = 4) -> List[Any]:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(func, self.documents.values()))

    def build_index(self, index_dir: str, schema: Optional[Schema] = None):
        if not WHOOSH_AVAILABLE:
            raise ImportError("Whoosh is not installed.")
        os.makedirs(index_dir, exist_ok=True)
        schema = schema or Schema(id=ID(stored=True), content=TEXT)
        ix = index.create_in(index_dir, schema)
        writer = ix.writer()
        for doc in self.documents.values():
            writer.add_document(id=doc.id, content=doc.text)
        writer.commit()

    def search(self, query: str, fuzzy: bool = False) -> List[str]:
        # Simple regex fallback
        if fuzzy:
            raise NotImplementedError("Fuzzy search requires an indexer.")
        pat = re.compile(query)
        return [doc.id for doc in self.documents.values() if pat.search(doc.text)]

    def version(self):
        raise NotImplementedError("Versioning/provenance not yet implemented.")

    def annotate(self, layer: str, annotations: Any):
        raise NotImplementedError("Annotation layers not yet implemented.")

    def align_sentences(self, other: 'Corpus', method: str = "fast_align"):
        raise NotImplementedError("Sentence alignment not yet implemented.")

    def link_entities(self, to: str = "wikidata"):
        raise NotImplementedError("Entity linking not yet implemented.")

    def extract_dates(self):
        raise NotImplementedError("Temporal extraction not yet implemented.")

    def extract_places(self):
        raise NotImplementedError("Geospatial extraction not yet implemented.")

    def filter_by_time(self, start: Any, end: Any):
        raise NotImplementedError("Time filtering not yet implemented.")

    def filter_by_region(self, bbox: Tuple[float,float,float,float]):
        raise NotImplementedError("Spatial filtering not yet implemented.")

    def compare(self, other: 'Corpus', metric: str = "jaccard"):
        raise NotImplementedError("Cross-corpus comparison not yet implemented.")

    def add_documents(self, docs: List[Document]):
        for d in docs:
            self.documents[d.id] = d

    def remove_documents(self, ids: List[str]):
        for i in ids:
            self.documents.pop(i, None)

    def annotate_with_spacy(self, model: str = "en_core_web_sm"):
        raise NotImplementedError("spaCy integration not yet implemented.")

    def apply_transformers(self, model: str = "bert-base-multilingual-cased"):
        raise NotImplementedError("Transformers integration not yet implemented.")

    def register_loader(self, name: str, fn: Callable):
        setattr(self.__class__, f"from_{name}", classmethod(fn))

    def fetch_gutenberg(self, ids: List[int]):
        raise NotImplementedError("Gutenberg fetching not yet implemented.")

    def fetch_hathitrust(self, collection: str):
        raise NotImplementedError("HathiTrust fetching not yet implemented.")

    def ocr_ingest(self, image_paths: List[str]):
        raise NotImplementedError("OCR ingestion not yet implemented.")

    def filter_by_license(self, license: str):
        return self.filter_by_metadata(license=license)

    def summary_report(self):
        raise NotImplementedError("HTML summary reports not yet implemented.")

    def plot_doc_length(self):
        raise NotImplementedError("Plotting utilities not yet implemented.")
