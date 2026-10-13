"""BM25 evidence retrieval over the FActScore Wikipedia snapshot (prereg §3).

Okapi BM25 as implemented by SQLite FTS5 (``bm25()``, k1 = 1.2, b = 0.75) over
every passage of every page, two indexed columns (page title, passage text)
weighted 1.0 each, tokenizer ``unicode61 remove_diacritics 2``. The index is
contentless; a side table maps each FTS rowid to (page title, passage index),
and passage text is read back from the snapshot itself.

Query = entity name + " " + fact, tokenized like the index, minus a fixed
English stopword list, deduplicated, OR-joined. Top 10 by BM25 score; ties
broken by page title, then passage index, ascending.
"""

import sqlite3
import unicodedata

SNAPSHOT_SEPARATOR = "####SPECIAL####SEPARATOR####"
TOKENIZER = "unicode61 remove_diacritics 2"
COLUMN_WEIGHTS = (1.0, 1.0)
TOP_K = 10
INITIAL_POOL = 200

# NLTK's English stopword list (179 words), fixed here so the query cannot
# depend on an installed package version.
STOPWORDS = frozenset("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its
itself they them their theirs themselves what which who whom this that that'll
these those am is are was were be been being have has had having do does did
doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down in
out on off over under again further then once here there when where why how all
any both each few more most other some such no nor not only own same so than too
very s t can will just don don't should should've now d ll m o re ve y ain aren
aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven
haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't
shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split())


class RetrievalRefused(RuntimeError):
    """The index or the query cannot support a preregistered retrieval."""


def _is_token_char(ch):
    return unicodedata.category(ch)[0] in ("L", "N") or unicodedata.category(ch) == "Co"


def tokenize(text):
    """Approximates FTS5 unicode61 remove_diacritics 2: fold case and
    diacritics, split on anything that is not a letter, number or private-use
    character."""
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").lower()
    tokens, current = [], []
    for ch in folded:
        if _is_token_char(ch):
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def query_terms(entity, fact):
    """The preregistered query: name + fact, stopwords removed, first occurrence kept."""
    seen, terms = set(), []
    for token in tokenize(f"{entity} {fact}"):
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    if not terms:
        raise RetrievalRefused(f"empty query for {entity!r} / {fact!r}")
    return terms


def match_expression(terms):
    """Each term quoted, OR-joined. Terms are alphanumeric runs, so no quote escaping is needed."""
    return " OR ".join(f'"{t}"' for t in terms)


# ------------------------------------------------------------------ index

def create_index(connection):
    connection.executescript(f"""
        CREATE TABLE passage_map(
            rowid INTEGER PRIMARY KEY, title TEXT NOT NULL, pidx INTEGER NOT NULL);
        CREATE VIRTUAL TABLE passages USING fts5(
            title, text, content='', tokenize='{TOKENIZER}');
    """)


def passages_of(text):
    """(passage index, passage) for every non-empty passage, indices as in the snapshot split."""
    return [(i, p) for i, p in enumerate(text.split(SNAPSHOT_SEPARATOR)) if p.strip()]


def build_index(snapshot_path, index_path, batch=50_000, limit_pages=None, progress=None):
    """Index every passage of every snapshot page, in snapshot rowid order. Returns counts."""
    snap = sqlite3.connect(f"file:{snapshot_path}?mode=ro", uri=True)
    idx = sqlite3.connect(index_path)
    idx.execute("PRAGMA journal_mode=OFF")
    idx.execute("PRAGMA synchronous=OFF")
    create_index(idx)
    rowid = pages = 0
    maps, docs = [], []
    query = "SELECT title, text FROM documents ORDER BY rowid"
    if limit_pages:
        query += f" LIMIT {int(limit_pages)}"
    for title, text in snap.execute(query):
        pages += 1
        for pidx, passage in passages_of(text):
            rowid += 1
            maps.append((rowid, title, pidx))
            docs.append((rowid, title, passage))
        if len(docs) >= batch:
            _flush(idx, maps, docs)
            maps, docs = [], []
            if progress:
                progress(pages, rowid)
    _flush(idx, maps, docs)
    idx.execute("INSERT INTO passages(passages) VALUES('optimize')")
    idx.commit()
    idx.close()
    snap.close()
    return {"pages": pages, "passages": rowid}


def _flush(idx, maps, docs):
    if not docs:
        return
    idx.executemany("INSERT INTO passage_map VALUES (?, ?, ?)", maps)
    idx.executemany("INSERT INTO passages(rowid, title, text) VALUES (?, ?, ?)", docs)
    idx.commit()


# ------------------------------------------------------------------ retrieval

def retrieve(index, terms, k=TOP_K, pool=INITIAL_POOL):
    """Top-k (bm25, title, pidx), ties by title then pidx. Exact at the cutoff.

    FTS5 bm25() is lower-is-better. A pool larger than k is fetched and sorted
    with the tie-break; if the k-th score is tied with the last pooled score,
    the pool is enlarged until it is not, so the cutoff never splits a tie
    arbitrarily.
    """
    expr = match_expression(terms)
    weights = ", ".join(str(w) for w in COLUMN_WEIGHTS)
    while True:
        rows = index.execute(
            f"SELECT p.rowid, bm25(passages, {weights}) AS s, m.title, m.pidx "
            f"FROM passages p JOIN passage_map m ON m.rowid = p.rowid "
            f"WHERE passages MATCH ? ORDER BY s LIMIT ?", (expr, pool)).fetchall()
        ranked = sorted(((s, title, pidx) for _, s, title, pidx in rows),
                        key=lambda r: (r[0], r[1], r[2]))
        if len(rows) < pool or len(ranked) <= k or ranked[k - 1][0] < rows[-1][1]:
            return ranked[:k]
        pool *= 4


def passage_text(snapshot, title, pidx):
    row = snapshot.execute("SELECT text FROM documents WHERE title = ?", (title,)).fetchone()
    if row is None:
        raise RetrievalRefused(f"page {title!r} not in the snapshot")
    parts = row[0].split(SNAPSHOT_SEPARATOR)
    if not (0 <= pidx < len(parts)) or not parts[pidx].strip():
        raise RetrievalRefused(f"passage {pidx} of {title!r} not in the snapshot")
    return parts[pidx]


def document_text(title, passage):
    """The premise given to the NLI scorer (prereg §3)."""
    return f"{title}\n{passage}"
