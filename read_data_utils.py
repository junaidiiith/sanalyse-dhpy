import os
import re
from typing import List, Union
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from LughaatNLP import LughaatNLP
from tqdm.auto import tqdm
from conllu import parse_incr, TokenList
from datasets import load_dataset

import nltk
nltk.download('punkt_tab')

# --- Oracle function (assumed to exist elsewhere) ---
# def get_topic_name(words: List[str]) -> str:
#     """Returns a human-readable name for a topic given its top words."""
#     ...


# make sure you’ve run: nltk.download('punkt'); nltk.download('stopwords')

def get_urdu_stopwords() -> List[str]:
    """Returns a list of Urdu stopwords."""
    # Assuming LughaatNLP has a method to get Urdu stopwords
    urdu_text_processing = LughaatNLP()
    return urdu_text_processing.stopwords_data['stopwords']


def nltk_simple_preprocess(text, min_len=2, max_len=15):
    # lowercase and tokenize
    tokens = word_tokenize(text.lower())
    # keep alphabetic tokens of desired length
    return [
        tok for tok in tokens
        if tok.isalpha()
        and min_len <= len(tok) <= max_len
    ]


def preprocess_urdu_doc(doc: str) -> List[str]:
    """Tokenize, lowercase, remove stopwords and short tokens for Urdu."""
    # Assuming a similar preprocessing function exists for Urdu
    urdu_text_processing = LughaatNLP()
    normalized_text = urdu_text_processing.normalize(doc)
    lemmatized_sentence = urdu_text_processing.lemmatize_sentence(normalized_text)
    stemmed_sentence = urdu_text_processing.urdu_stemmer(lemmatized_sentence)
    filtered_text = urdu_text_processing.remove_stopwords(stemmed_sentence)
    tokenized_text = urdu_text_processing.urdu_tokenize(filtered_text)
    
    return tokenized_text


def preprocess(doc: str, language='en') -> List[str]:
    """Tokenize, lowercase, remove stopwords and short tokens."""
    if language == 'ur':
        return preprocess_urdu_doc(doc)
    STOPWORDS = set(stopwords.words('english'))
    return [
        token
        for token in nltk_simple_preprocess(doc)
        if token not in STOPWORDS and len(token) > 3
    ]




def read_conllu(file_path: str) -> List[TokenList]:
    """
    Reads a CoNLL-U file and returns a list of conllu.TokenList objects,
    one per sentence. Each TokenList behaves like a list of dicts, where
    each dict has keys: 'id', 'form', 'lemma', 'upos', 'xpos', 'feats',
    'head', 'deprel', 'deps', and 'misc'.
    
    Args:
        file_path: path to the .conllu file
        
    Returns:
        List of TokenList, each representing one sentence.
    """
    docs: List[TokenList] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for token_list in parse_incr(f):
            docs.append(token_list)
    return docs


def get_conll_doc_texts(conllu_path: str) -> List[str]:
    """
    Reads a CoNLL-U file and returns the text of each sentence.
    
    - If the CoNLL-U has a metadata line `# text = …`, that exact string is used.
    - Otherwise, tokens are joined by spaces to reconstruct the sentence.
    """
    docs = read_conllu(conllu_path)
    texts = []
    for doc in docs:
        # Check for metadata line first
        if 'text' in doc.metadata:
            texts.append(doc.metadata['text'])
        else:
            # Join tokens by space if no metadata text
            texts.append(" ".join(token['form'] for token in doc))
    return texts


def get_language_conllu_code(language: str) -> str:
    """
    Returns the CoNLL-U language code for a given language.
    
    Args:
    """
    if language == "ur":
        language = "ur_udtb"
    elif language == "en":
        language = "en_ewt"
    elif language == "fr":
        language = "fr_gsd"
    elif language == "de":
        language = "de_gsd"
    elif language == "es":
        language = "es_gsd"
    elif language == "ar":
        language = "ar_padt"
    elif language == "hi":
        language = "hi_hdtb"
    elif language == "zh":
        language = "zh_gsd"
    elif language == "ru":
        language = "ru_syntagrus"
    elif language == "ja":
        language = "ja_gsd"
    elif language == "ko":
        language = "ko_gsd"
    elif language == "it":
        language = "it_isdt"
    elif language == "pt":
        language = "pt_bosque"
    elif language == "tr":
        language = "tr_imst"
    else:
        raise ValueError(f"Unsupported language: {language}. Please provide a valid language code.")
    return language


def load_conll_data(language: str = "en") -> List[str]:
    """Loads CoNLL data for a given language."""
    # Load other languages' CoNLL data
    # Note: You can specify other language codes as needed
    # e.g., "en_ewt" for English, "fr_gsd"
    # or any other language supported by Universal Dependencies
    # For example, "en_ewt" for English EWT treebank
    # or "fr_gsd" for French GSD treebank
    # Adjust the language code as needed
    ud = load_dataset("universal_dependencies", get_language_conllu_code(language))

    splits = {
        "train": ud["train"],
        "validation": ud.get("validation", ud.get("dev")),
        "test": ud["test"],
    }
    
    return sum([d["text"] for d in splits.values()], [])


def read_folder_files(
    folder_path: str,
    file_extension: str = ".txt",
    threshold: float = 0.7
) -> List[str]:
    """Reads all files with a given extension from a folder."""

    # compile once: Unicode ranges covering Urdu/Arabic script
    URDU_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')

    def is_mostly_urdu(line: str, threshold: float = 0.7) -> bool:
        """
        Check if the majority of non-whitespace characters in `line` are Urdu.
        
        Args:
            line:    input string (one line of text)
            threshold: fraction (0-1) above which we consider it "mostly Urdu"
                    (default 0.5)
        
        Returns:
            True if (Urdu_chars / non_space_chars) > threshold.
        """
        # count non-whitespace chars
        non_space_chars = [c for c in line if not c.isspace()]
        total = len(non_space_chars)
        if total == 0:
            return False
        
        # count Urdu script chars
        urdu_count = sum(bool(URDU_RE.match(c)) for c in non_space_chars)
        
        return (urdu_count / total) > threshold

    
    documents = []
    for filename in os.listdir(folder_path):
        if file_extension in ['.conll', '.conllu'] and filename.endswith(file_extension):
            sentences = get_conll_doc_texts(os.path.join(folder_path, filename))
            documents.extend(sentences)
        
        elif filename.endswith(file_extension):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                doc = f.read().replace("www.emarsiya.com", "")
                doc = "\n\n".join(
                    "\n".join([line for line in paragraph.split("\n") if is_mostly_urdu(line, threshold=threshold)]) 
                    for paragraph in doc.split("\n\n")
                )
                documents.append(doc.strip())
    return documents


def get_documents_data(
    docs_path: str = None, 
    file_extension: str = ".conllu",
    threshold: float = 0.7,
    use_ud: Union[bool, None] = None, 
    language='en'
):
    assert use_ud in [None, True, False], "use_ud must be None, True, or False"
    assert use_ud is not None or isinstance(docs_path, str), "If use_ud is None, docs_path must be a string path to a folder."
    if use_ud:
        # Load CoNLL data for Urdu
        documents = load_conll_data(language=language)
    elif isinstance(docs_path, str):
        documents = read_folder_files(docs_path, file_extension=file_extension, threshold=threshold)
    else:
        raise ValueError("docs_path must be a string path to a folder if use_ud is None.")
    
    print(f"Loaded {len(documents)} documents from {docs_path} with extension {file_extension}.")
    assert len(documents) > 0, "No documents loaded. Check the path and file extension."
    
    texts = [preprocess(doc, language) for doc in tqdm(documents, desc="Preprocessing documents")]
    
    return texts