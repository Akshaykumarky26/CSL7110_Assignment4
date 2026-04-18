"""
Part 2: Inverted Index for Web Search.
Classes: MySet, Position, WordEntry, PageIndex, PageEntry, MyHashTable,
         InvertedPageIndex, SearchEngine.

Rules from assignment:
- Convert each word to lowercase.
- Do not store connector words, but DO count their indices.
  Connector words: { a, an, the, they, these, this, for, is, are, was, of, or,
                     and, does, will, whose }.
- Replace punctuation with a space:
    { } [ ] < > = ( ) . , ; ' " ? # ! - :
- Treat plural and singular forms as the same:
    (stack, stacks), (structure, structures), (application, applications).

Note on class API:
  For practical file loading, PageEntry and SearchEngine are parameterized with
  folder paths (PageEntry(pageName, folder), SearchEngine(folder)). This does
  not change the required behaviour.

Usage:
    python3 part2_websearch.py
    (expects actions.txt, answers.txt and webpages/ in the same folder
     as this script, unless WEBSEARCH_DIR env var overrides)
"""
import os
import math

# ---------- Constants ----------
CONNECTOR_WORDS = {
    "a", "an", "the", "they", "these", "this", "for", "is",
    "are", "was", "of", "or", "and", "does", "will", "whose",
}

PUNCTUATION = set(list("{}[]<>=().,;'\"?#!-:"))

PLURAL_TO_SINGULAR = {
    "stacks": "stack",
    "structures": "structure",
    "applications": "application",
}


def normalize_word(w):
    w = w.lower()
    return PLURAL_TO_SINGULAR.get(w, w)


def tokenize(text):
    chars = [(" " if ch in PUNCTUATION else ch) for ch in text]
    return [normalize_word(t) for t in "".join(chars).split() if t.strip()]


# ---------- MySet ----------
class MySet:
    def __init__(self, iterable=None):
        self._data = set()
        if iterable is not None:
            for x in iterable:
                self._data.add(x)
    def addElement(self, element): self._data.add(element)
    def union(self, other):        r = MySet(); r._data = self._data | other._data; return r
    def intersection(self, other): r = MySet(); r._data = self._data & other._data; return r
    def __iter__(self):            return iter(self._data)
    def __len__(self):             return len(self._data)
    def __contains__(self, x):     return x in self._data


# ---------- Position ----------
class Position:
    def __init__(self, p, wordIndex):
        self._p, self._w = p, wordIndex
    def getPageEntry(self): return self._p
    def getWordIndex(self): return self._w


# ---------- WordEntry ----------
class WordEntry:
    def __init__(self, word):
        self._word, self._positions = word, []
    def addPosition(self, position):   self._positions.append(position)
    def addPositions(self, positions): self._positions.extend(positions)
    def getAllPositionsForThisWord(self): return list(self._positions)
    def getWord(self): return self._word
    def getTermFrequency(self, page):
        count = sum(1 for pos in self._positions if pos.getPageEntry() is page)
        total = page.getTotalWords()
        return 0.0 if total == 0 else count / total


# ---------- PageIndex ----------
class PageIndex:
    def __init__(self):
        self._entries = {}
    def addPositionForWord(self, s, position):
        s = normalize_word(s)
        if s in self._entries:
            self._entries[s].addPosition(position)
        else:
            we = WordEntry(s); we.addPosition(position); self._entries[s] = we
    def getWordEntries(self): return list(self._entries.values())
    def getWordEntry(self, s): return self._entries.get(normalize_word(s))


# ---------- PageEntry ----------
class PageEntry:
    def __init__(self, pageName, folder):
        self._pageName = pageName
        self._pageIndex = PageIndex()
        with open(os.path.join(folder, pageName), "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenize(text)
        self._totalWords = len(tokens)
        for idx, tok in enumerate(tokens, start=1):  # 1-based indices
            if tok in CONNECTOR_WORDS:
                continue  # skip stop words (but idx still advances)
            self._pageIndex.addPositionForWord(tok, Position(self, idx))
    def getPageIndex(self): return self._pageIndex
    def getPageName(self): return self._pageName
    def getTotalWords(self): return self._totalWords


# ---------- MyHashTable ----------
class MyHashTable:
    def __init__(self, size=1024):
        self._size = size
        self._buckets = {}
    def getHashIndex(self, s): return hash(normalize_word(s)) % self._size
    def addPositionsForWord(self, w):
        key = w.getWord()
        if key in self._buckets:
            self._buckets[key].addPositions(w.getAllPositionsForThisWord())
        else:
            new_we = WordEntry(key); new_we.addPositions(w.getAllPositionsForThisWord())
            self._buckets[key] = new_we
    def get(self, s):      return self._buckets.get(normalize_word(s))
    def contains(self, s): return normalize_word(s) in self._buckets


# ---------- InvertedPageIndex ----------
class InvertedPageIndex:
    def __init__(self):
        self._table = MyHashTable()
        self._pages = {}
    def addPage(self, p):
        self._pages[p.getPageName()] = p
        for we in p.getPageIndex().getWordEntries():
            self._table.addPositionsForWord(we)
    def getPagesWhichContainWord(self, s):
        s = normalize_word(s)
        out = MySet()
        we = self._table.get(s)
        if we is None: return out
        for pos in we.getAllPositionsForThisWord():
            out.addElement(pos.getPageEntry())
        return out
    def getPage(self, name):  return self._pages.get(name)
    def hasPage(self, name):  return name in self._pages


# ---------- SearchEngine ----------
class SearchEngine:
    def __init__(self, folder):
        self._invertedIndex = InvertedPageIndex()
        self._folder = folder
        self._outputs = []
    def _emit(self, s):
        print(s)
        self._outputs.append(s)
    def _addPage(self, pageName):
        try:
            p = PageEntry(pageName, self._folder)
        except FileNotFoundError:
            self._emit(f"No file {pageName} found"); return
        self._invertedIndex.addPage(p)
    def _queryFindPagesWhichContainWord(self, x):
        pages = self._invertedIndex.getPagesWhichContainWord(x)
        if len(pages) == 0:
            self._emit(f"No webpage contains word {x}")
        else:
            self._emit(", ".join(sorted(p.getPageName() for p in pages)))
    def _queryFindPositionsOfWordInAPage(self, x, y):
        if not self._invertedIndex.hasPage(y):
            self._emit(f"No webpage {y} found"); return
        page = self._invertedIndex.getPage(y)
        we = page.getPageIndex().getWordEntry(x)
        if we is None or not we.getAllPositionsForThisWord():
            self._emit(f"Webpage {y} does not contain word {x}"); return
        self._emit(", ".join(str(pos.getWordIndex())
                             for pos in we.getAllPositionsForThisWord()))
    def performAction(self, actionMessage):
        parts = actionMessage.strip().split()
        if not parts: return
        cmd = parts[0]
        if cmd == "addPage" and len(parts) == 2:
            self._addPage(parts[1])
        elif cmd == "queryFindPagesWhichContainWord" and len(parts) == 2:
            self._queryFindPagesWhichContainWord(parts[1])
        elif cmd == "queryFindPositionsOfWordInAPage" and len(parts) == 3:
            self._queryFindPositionsOfWordInAPage(parts[1], parts[2])
        else:
            self._emit(f"Unknown action: {actionMessage}")

    def tfidfScore(self, word, pageName):
        """Return tf(word, page) * idf(word) for the collection."""
        page = self._invertedIndex.getPage(pageName)
        if page is None: return 0.0
        we = page.getPageIndex().getWordEntry(word)
        if we is None: return 0.0
        tf = we.getTermFrequency(page)
        N = len(self._invertedIndex._pages)
        n_w = len(self._invertedIndex.getPagesWhichContainWord(word))
        if n_w == 0 or N == 0: return 0.0
        return tf * math.log(N / n_w)


def run_actions(actions_file, folder):
    se = SearchEngine(folder)
    with open(actions_file, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                se.performAction(ln)
    return se._outputs


if __name__ == "__main__":
    BASE = os.environ.get("WEBSEARCH_DIR",
                          os.path.dirname(os.path.abspath(__file__)))
    actions = os.path.join(BASE, "actions.txt")
    answers = os.path.join(BASE, "answers.txt")
    folder = os.path.join(BASE, "webpages")

    outputs = run_actions(actions, folder)

    print("\n--- Compare with answers.txt ---")
    with open(answers, "r", encoding="utf-8") as f:
        expected = [ln.rstrip("\n") for ln in f if ln.strip()]
    all_ok = True
    for i, (got, exp) in enumerate(zip(outputs, expected)):
        ok = got.strip() == exp.strip()
        all_ok &= ok
        print(f"{'OK ' if ok else 'DIFF'} line {i+1}: got={got!r}  exp={exp!r}")
    print("\nALL PASSED" if all_ok else "\nSOME FAILURES")
