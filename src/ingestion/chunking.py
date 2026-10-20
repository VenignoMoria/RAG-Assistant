import hashlib
import re
from html import unescape
from html.parser import HTMLParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from copilot.domain import Chunk, KnowledgeEvent


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style"}:
            self.skip += 1
        elif tag in {"p", "br", "div", "li", "h1", "h2"} and not self.skip:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def chunks_for(event: KnowledgeEvent) -> list[Chunk]:
    text = event.content
    if event.content_type == "text/html":
        extractor = TextExtractor()
        extractor.feed(text)
        text = " ".join(extractor.parts)
    text = re.sub(r"[ \t]+", " ", unescape(text)).strip()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120,
                                             separators=["\n\n", "\n", ". ", " ", ""])
    chunks = []
    for index, part in enumerate(splitter.split_text(text)):
        digest = hashlib.sha256(part.encode()).hexdigest()[:16]
        chunks.append(Chunk(chunk_id=f"{event.document_id}:{event.version}:{index}:{digest}",
                            document_id=event.document_id, version=event.version, title=event.title,
                            text=part, vertical=event.vertical, public=event.public, source=event.source))
    if not chunks:
        raise ValueError("Document has no text")
    return chunks
