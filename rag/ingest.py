"""Load movie records, extract metadata, and build retrieval chunks."""

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


SECTION_ALIASES = {
    "plot summary": "plot",
    "synopsis": "plot",
    "overview": "overview",
    "cast": "cast",
    "cast (subjects)": "cast",
    "awards": "awards",
    "awards and reception": "awards",
    "reception": "reception",
    "production": "production",
    "source": "sources",
    "sources": "sources",
}

INLINE_FIELDS = {
    "year",
    "director",
    "genre",
    "country",
    "language",
    "runtime",
    "budget",
    "technique",
}


@dataclass
class MovieMetadata:
    """Normalized, searchable metadata derived from one source document."""

    movie_id: str
    source_file: str
    title: str
    display_title: str
    alternate_titles: List[str] = field(default_factory=list)
    year: Optional[int] = None
    directors: List[str] = field(default_factory=list)
    genres: List[str] = field(default_factory=list)
    countries: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    runtime_minutes: Optional[int] = None
    cast: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    plot: str = ""
    overview: str = ""
    awards: str = ""
    reception: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str
    movie_id: str = ""
    metadata: Dict = field(default_factory=dict)


def _split_values(value: str) -> List[str]:
    return [part.strip() for part in re.split(r",|;", value) if part.strip()]


def _parse_directors(value: str) -> List[str]:
    """Remove common prose appended to director fields, then split co-directors."""
    value = re.sub(r"^(?:Khmer novelist|Chinese Cambodian director),?\s*", "", value)
    value = re.split(
        r"\s+(?:and written by|and starring|and stars|and actress|starring|which explores|"
        r"who was|who experienced|based on|following|and is the|and was produced|"
        r"in (?:his|her|their) (?:feature film )?directorial debut(?:s)?)\b|"
        r",\s+(?:a|the)\s+",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.sub(r"['’]s debut film.*$", "", value, flags=re.IGNORECASE).strip(" ,")
    return [name.strip() for name in re.split(r"\s+and\s+|\s*&\s*", value) if name.strip()]


def _canonical_title(display_title: str) -> str:
    title = re.sub(r"\s*\(film\)", "", display_title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(\d{4}\s+film\)", "", title, flags=re.IGNORECASE)
    title = re.split(r"\s+\((?:Khmer|French):", title, maxsplit=1, flags=re.IGNORECASE)[0]
    return title.strip()


def _parse_sections(lines: List[str]) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current = "preamble"
    sections[current] = []

    for line in lines:
        stripped = line.strip()
        match = re.fullmatch(r"([^:]{2,50}):", stripped)
        alias = SECTION_ALIASES.get(match.group(1).lower()) if match else None
        if alias:
            current = alias
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(stripped)

    return {
        name: "\n".join(part for part in parts if part).strip()
        for name, parts in sections.items()
    }


def parse_movie_document(filename: str, text: str) -> dict:
    """Parse a legacy movie text file into raw text plus normalized metadata."""
    lines = text.strip().splitlines()
    display_title = next((line.strip() for line in lines if line.strip()), "")
    title = _canonical_title(display_title)
    movie_id = os.path.splitext(os.path.basename(filename))[0].lower().replace("_", "-")

    fields: Dict[str, str] = {}
    for line in lines[1:]:
        match = re.match(r"^([A-Za-z]+):\s*(.+)$", line.strip())
        if match and match.group(1).lower() in INLINE_FIELDS:
            fields[match.group(1).lower()] = match.group(2).strip()

    sections = _parse_sections(lines[1:])

    year = None
    year_match = re.search(r"\b(18|19|20)\d{2}\b", fields.get("year", ""))
    if not year_match:
        year_match = re.search(r"\b(18|19|20)\d{2}\b", text)
    if year_match:
        year = int(year_match.group(0))

    director_text = fields.get("director", "")
    if not director_text:
        director_match = re.search(r"\bdirected (?:and co-written )?by ([^.\n]+)", text, re.IGNORECASE)
        director_text = director_match.group(1).strip() if director_match else ""

    runtime_match = re.search(r"(\d+)\s*(?:minutes?|mins?)", fields.get("runtime", ""), re.I)
    cast_lines = sections.get("cast", "").splitlines()
    cast = []
    for line in cast_lines:
        name = re.split(r"\s+as\s+", line, maxsplit=1, flags=re.IGNORECASE)[0].strip(" -")
        if name and not name.startswith("http"):
            cast.append(name)

    source_urls = re.findall(r"https?://[^\s]+", sections.get("sources", ""))
    alternate_titles = [display_title] if display_title and display_title != title else []

    metadata = MovieMetadata(
        movie_id=movie_id,
        source_file=os.path.basename(filename),
        title=title,
        display_title=display_title or title,
        alternate_titles=alternate_titles,
        year=year,
        directors=_parse_directors(director_text) if director_text else [],
        genres=_split_values(fields.get("genre", "")),
        countries=_split_values(fields.get("country", "")),
        languages=_split_values(fields.get("language", "")),
        runtime_minutes=int(runtime_match.group(1)) if runtime_match else None,
        cast=cast,
        source_urls=source_urls,
        plot=sections.get("plot", ""),
        overview=sections.get("overview", ""),
        awards=sections.get("awards", ""),
        reception=sections.get("reception", ""),
    )
    return {
        "id": movie_id,
        "title": title,
        "text": text.strip(),
        "metadata": asdict(metadata),
    }


def load_documents(folder: str) -> List[dict]:
    """Load and normalize every ``.txt`` movie document in a folder."""
    docs = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as source:
            text = source.read().strip()
        if text:
            docs.append(parse_movie_document(filename, text))
    return docs


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Split text into overlapping word-count chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def _searchable_prefix(metadata: Dict) -> str:
    fields = [
        metadata.get("title", ""),
        str(metadata.get("year") or ""),
        " ".join(metadata.get("directors", [])),
        " ".join(metadata.get("genres", [])),
        " ".join(metadata.get("countries", [])),
        " ".join(metadata.get("languages", [])),
    ]
    return " | ".join(value for value in fields if value)


def build_chunk_records(
    docs: List[dict], chunk_size: int = 80, overlap: int = 20
) -> List[Chunk]:
    """Build chunks that retain their movie metadata and searchable context."""
    records = []
    for doc in docs:
        metadata = doc.get("metadata", {})
        prefix = _searchable_prefix(metadata)
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for index, piece in enumerate(pieces):
            records.append(
                Chunk(
                    chunk_id=f"{doc['id']}::{index}",
                    movie_id=doc["id"],
                    doc_title=doc["title"],
                    text=f"{prefix}\n{piece}" if prefix else piece,
                    metadata=metadata,
                )
            )
    return records
