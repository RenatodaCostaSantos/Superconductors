"""Minimal pure-Python reader for R workspace files (.RData, XDR format version 2).

Written to extract the element tables from the author's ``tc.RData`` without installing R. It follows
R's ``src/main/serialize.c`` (``ReadItem`` / ``ReadBC``): every object in the file is walked so the
reader can skip past functions, byte code and the xgboost model, but only vectors, lists and
data frames are converted to Python. Existing Python readers (pyreadr, rdata) stop on this file
because of its byte-compiled functions and raw vectors.
"""

import gzip
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# SEXP type codes (R Internals, "Serialization Formats")
NILSXP, SYMSXP, LISTSXP, CLOSXP, ENVSXP, PROMSXP, LANGSXP = 0, 1, 2, 3, 4, 5, 6
SPECIALSXP, BUILTINSXP, CHARSXP, LGLSXP, INTSXP, REALSXP, CPLXSXP = 7, 8, 9, 10, 13, 14, 15
STRSXP, DOTSXP, VECSXP, EXPRSXP, BCODESXP, EXTPTRSXP, WEAKREFSXP, RAWSXP, S4SXP = (
    16, 17, 19, 20, 21, 22, 23, 24, 25)
ALTREP_SXP, ATTRLISTSXP, ATTRLANGSXP = 238, 239, 240
BASEENV_SXP, EMPTYENV_SXP, BCREPREF, BCREPDEF = 241, 242, 243, 244
PERSISTSXP, PACKAGESXP, NAMESPACESXP, BASENAMESPACE_SXP = 247, 248, 249, 250
MISSINGARG_SXP, UNBOUNDVALUE_SXP, GLOBALENV_SXP, NILVALUE_SXP, REFSXP = 251, 252, 253, 254, 255
NO_DATA = {NILVALUE_SXP, GLOBALENV_SXP, UNBOUNDVALUE_SXP, MISSINGARG_SXP, BASENAMESPACE_SXP,
           EMPTYENV_SXP, BASEENV_SXP}
PAIRLISTS = {LISTSXP, CLOSXP, PROMSXP, LANGSXP, DOTSXP, ATTRLISTSXP, ATTRLANGSXP}
R_NA_INTEGER = -(2**31)


class RVector:
    """An R vector: its type code, values (None when skipped) and attributes (dict)."""

    def __init__(self, sexptype, values, attributes):
        self.sexptype, self.values, self.attributes = sexptype, values, attributes


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos, self.refs = data, 0, []

    def int(self) -> int:
        (value,) = struct.unpack_from(">i", self.data, self.pos)
        self.pos += 4
        return value

    def bytes(self, n: int) -> bytes:
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def item(self):
        flags = self.int()
        sexptype = flags & 0xFF
        has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))

        if sexptype == REFSXP:
            index = flags >> 8
            return self.refs[(index or self.int()) - 1]
        if sexptype in NO_DATA:
            return None
        if sexptype in (PERSISTSXP, PACKAGESXP, NAMESPACESXP):
            self.int()  # always 0
            value = ("namespace", [self.item() for _ in range(self.int())])
            self.refs.append(value)
            return value
        if sexptype == SYMSXP:
            name = self.item()
            self.refs.append(name)
            return name
        if sexptype == ALTREP_SXP:
            self.item(), self.item(), self.item()  # class info, state, attributes
            return None
        if sexptype in PAIRLISTS:
            attributes = self.item() if has_attr else None
            tag = self.item() if has_tag else None
            car, cdr = self.item(), self.item()
            return ("pairlist", tag, car, cdr, attributes)
        if sexptype == ENVSXP:
            self.refs.append("environment")
            self.int()  # locked
            self.item(), self.item(), self.item(), self.item()  # enclosure, frame, hash table, attributes
            return "environment"
        if sexptype == CHARSXP:
            n = self.int()
            return None if n == -1 else self.bytes(n).decode("utf-8", "replace")

        if sexptype == EXTPTRSXP:
            self.refs.append("external pointer")
            self.item(), self.item()  # protected value, tag
            values = None
        elif sexptype == WEAKREFSXP:
            self.refs.append("weak reference")
            values = None
        elif sexptype in (SPECIALSXP, BUILTINSXP):
            values = self.bytes(self.int()).decode()
        elif sexptype == BCODESXP:
            self._bytecode([None] * self.int())
            values = None
        elif sexptype == S4SXP:
            values = None
        elif sexptype in (LGLSXP, INTSXP, REALSXP, CPLXSXP, STRSXP, VECSXP, EXPRSXP, RAWSXP):
            n = self.int()
            if n == -1:  # long vector
                n = (self.int() << 32) + self.int()
            if sexptype in (LGLSXP, INTSXP):
                values = np.frombuffer(self.bytes(4 * n), dtype=">i4").astype(np.int64)
            elif sexptype == REALSXP:
                values = np.frombuffer(self.bytes(8 * n), dtype=">f8").astype(np.float64)
            elif sexptype == CPLXSXP:
                self.bytes(16 * n)
                values = None
            elif sexptype == RAWSXP:
                self.bytes(n)
                values = None
            else:
                values = [self.item() for _ in range(n)]
        else:
            raise ValueError(f"Unsupported R object type {sexptype} at byte {self.pos}")

        attributes = _pairlist_to_dict(self.item()) if has_attr else {}
        return RVector(sexptype, values, attributes)

    def _bytecode(self, reps):
        self.item()  # code
        for _ in range(self.int()):
            sexptype = self.int()
            if sexptype == BCODESXP:
                self._bytecode(reps)
            elif sexptype in (LANGSXP, LISTSXP, BCREPDEF, BCREPREF, ATTRLANGSXP, ATTRLISTSXP):
                self._bytecode_lang(sexptype, reps)
            else:
                self.item()

    def _bytecode_lang(self, sexptype, reps):
        if sexptype == BCREPREF:
            return reps[self.int()]
        if sexptype in (BCREPDEF, LANGSXP, LISTSXP, ATTRLANGSXP, ATTRLISTSXP):
            position = -1
            if sexptype == BCREPDEF:
                position, sexptype = self.int(), self.int()
            if position >= 0:
                reps[position] = "language"
            if sexptype in (ATTRLANGSXP, ATTRLISTSXP):
                self.item()  # attributes
            self.item()  # tag
            self._bytecode_lang(self.int(), reps)  # car
            self._bytecode_lang(self.int(), reps)  # cdr
            return "language"
        return self.item()


def _pairlist_to_dict(node) -> dict:
    out = {}
    while isinstance(node, tuple) and node[0] == "pairlist":
        _, tag, car, cdr, _ = node
        out[tag] = car
        node = cdr
    return out


def read_rdata(path: str | Path) -> dict[str, object]:
    """Return {object name: parsed object} for every object saved in an .RData file."""
    data = gzip.open(path).read()
    if data[:7] != b"RDX2\nX\n":
        raise ValueError(f"{path} is not an XDR (version 2) R workspace")
    reader = _Reader(data)
    reader.pos = 7
    reader.int(), reader.int(), reader.int()  # format version, writer R version, minimal reader version

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 100_000))  # long pairlists are nested in the file
    try:
        objects = {}
        # the workspace is a pairlist of (name, object); walk it iteratively
        while True:
            flags = reader.int()
            if flags & 0xFF == NILVALUE_SXP:
                return objects
            if flags & (1 << 9):
                reader.item()
            name = reader.item()
            objects[name] = reader.item()
    finally:
        sys.setrecursionlimit(limit)


def _column(vector: RVector):
    values, attributes = vector.values, vector.attributes
    if "levels" in attributes:  # factor: integer codes into the levels
        levels = attributes["levels"].values
        return [None if code == R_NA_INTEGER else levels[code - 1] for code in values]
    if vector.sexptype in (INTSXP, LGLSXP):
        return np.where(values == R_NA_INTEGER, np.nan, values) if (values == R_NA_INTEGER).any() else values
    return values


def to_dataframe(vector: RVector) -> pd.DataFrame:
    """Convert a parsed R data.frame to pandas (factors become strings, NA becomes NaN/None)."""
    attributes = vector.attributes
    if attributes.get("class") is None or "data.frame" not in attributes["class"].values:
        raise TypeError("not an R data.frame")
    names = attributes["names"].values
    df = pd.DataFrame({name: _column(col) for name, col in zip(names, vector.values)})
    row_names = attributes.get("row.names")
    if row_names is not None and row_names.sexptype == STRSXP:
        df.index = row_names.values
    return df
