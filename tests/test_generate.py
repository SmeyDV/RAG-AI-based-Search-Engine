import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.generate import _load_env_file


class EnvironmentFileTests(unittest.TestCase):
    def test_loads_key_from_env_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text('DEEPSEEK_API_KEY="test-key"\n', encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                _load_env_file(path)
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "test-key")

    def test_existing_environment_value_takes_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text("DEEPSEEK_API_KEY=file-key\n", encoding="utf-8")
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "server-key"}, clear=True):
                _load_env_file(path)
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "server-key")


if __name__ == "__main__":
    unittest.main()
