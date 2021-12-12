from pathlib import Path

from src import utils


class Multimedia:
    MULTIMEDIA_PATH = utils.get_data_path('multimedia')
    MULTIMEDIA_KEYWORDS = ("pictur", "imag", "poster", "frame")

    def __init__(self, path: Path = MULTIMEDIA_PATH):
        self._read_data(path)

    def _read_data(self, path: Path) -> None:
        pass

    def process_question(self, question: str) -> str:
        if "poster" in question:
            return self._process_poster(question)
        elif "frame" in question:
            return self._process_frame(question)
        else:
            return self._process_picture(question)

    def _process_poster(self, question: str) -> str:
        pass

    def _process_frame(self, question: str) -> str:
        pass

    def _process_picture(self, question: str) -> str:
        pass
