import unittest

from rag.ingest import build_chunk_records, chunk_text, parse_movie_document


SAMPLE = """Example Film (Khmer: ឧទាហរណ៍)

Year: 2024
Director: Dara One and Dara Two
Genre: Drama, Mystery
Country: Cambodia, France
Language: Khmer
Runtime: 95 minutes

Cast:
Actor One as Hero
Actor Two as Friend

Plot Summary:
A short example plot.

Awards:
An example award.

Sources:
https://example.com/movie
"""


class MovieParsingTests(unittest.TestCase):
    def test_structured_metadata(self):
        document = parse_movie_document("Example_Film.txt", SAMPLE)
        metadata = document["metadata"]

        self.assertEqual(document["id"], "example-film")
        self.assertEqual(document["title"], "Example Film")
        self.assertEqual(metadata["source_file"], "Example_Film.txt")
        self.assertEqual(metadata["year"], 2024)
        self.assertEqual(metadata["directors"], ["Dara One", "Dara Two"])
        self.assertEqual(metadata["genres"], ["Drama", "Mystery"])
        self.assertEqual(metadata["countries"], ["Cambodia", "France"])
        self.assertEqual(metadata["runtime_minutes"], 95)
        self.assertEqual(metadata["cast"], ["Actor One", "Actor Two"])
        self.assertEqual(metadata["source_urls"], ["https://example.com/movie"])

    def test_chunks_retain_metadata(self):
        document = parse_movie_document("Example_Film.txt", SAMPLE)
        chunk = build_chunk_records([document], chunk_size=30, overlap=5)[0]

        self.assertEqual(chunk.movie_id, "example-film")
        self.assertEqual(chunk.metadata["year"], 2024)
        self.assertIn("Dara One", chunk.text)

    def test_chunk_arguments_are_validated(self):
        with self.assertRaises(ValueError):
            chunk_text("some text", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text("some text", chunk_size=10, overlap=10)

    def test_director_prose_is_not_treated_as_a_name(self):
        text = """Legacy Movie

Year: 1968
Director: Yvon Hem and stars Actor One and Actor Two
"""
        document = parse_movie_document("Legacy_Movie.txt", text)
        self.assertEqual(document["metadata"]["directors"], ["Yvon Hem"])


if __name__ == "__main__":
    unittest.main()
