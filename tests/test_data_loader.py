import io

from src.data_loader import read_bytes_to_dataframe


def test_read_csv_bytes():
    df = read_bytes_to_dataframe(b'a,b\n1,2\n3,4\n', 'demo.csv')
    assert df.shape == (2, 2)


def test_read_json_bytes():
    df = read_bytes_to_dataframe(b'{"data":[{"a":1},{"a":2}]}', 'demo.json')
    assert df.shape == (2, 1)
