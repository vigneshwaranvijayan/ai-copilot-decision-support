import io
import json

from src.data_utils import load_tabular_file


class NamedBytesIO(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def test_load_csv_file():
    f = NamedBytesIO(b"a,b\n1,x\n2,y\n", "sample.csv")
    df = load_tabular_file(f)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_load_json_records_file():
    payload = json.dumps([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]).encode("utf-8")
    f = NamedBytesIO(payload, "sample.json")
    df = load_tabular_file(f)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)
