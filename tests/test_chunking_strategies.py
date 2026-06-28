from src.chunking.chunking_strategies import SimpleChunker, AdvancedChunker


def test_simple_chunker():
    chunker = SimpleChunker(chunk_size=10, overlap=2)
    
    # Test empty string
    assert chunker.chunk_text("") == []
    
    # Test short string
    assert chunker.chunk_text("hello") == ["hello"]
    
    # Test exact size match
    assert chunker.chunk_text("1234567890") == ["1234567890"]
    
    # Test sliding window with overlap (chunk_size=10, overlap=2)
    # chunk 1: text[0:10] = "abcdefghij"
    # start shifts to 10 - 2 = 8
    # chunk 2: text[8:18] = "ijklmnopqr"
    # start shifts to 8 + 8 = 16
    # chunk 3: text[16:20] = "qrst"
    text = "abcdefghijklmnopqrst"
    chunks = chunker.chunk_text(text)
    assert chunks == ["abcdefghij", "ijklmnopqr", "qrst"]


def test_advanced_chunker():
    chunker = AdvancedChunker(chunk_size=20, overlap=5)
    
    # Test empty string and short string
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("hello") == ["hello"]
    
    # Test recursive splitter prefers paragraphs and sentences
    text = "Paragraph one.\n\nParagraph two is longer."
    # Since Paragraph one. (14 chars) + \n\n + Paragraph two is longer. (24 chars) is 40 chars total,
    # which is > 20, it splits on "\n\n" first, then further splits
    # "Paragraph two is longer." (24 chars) since it exceeds chunk_size=20
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 3
    assert chunks[0] == "Paragraph one.\n\n"
    assert "Paragraph two" in chunks[1]
    assert "longer." in chunks[2]
