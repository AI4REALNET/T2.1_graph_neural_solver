import pathlib
import json
import numpy

__version__ = "0.0.1"

here = pathlib.Path(__file__).parent.resolve()

def get_version(rel_path="__init__.py"):
    init_content = (here / rel_path).read_text(encoding='utf-8')
    for line in init_content.split('\n'):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


class NpEncoder(json.JSONEncoder):
    """
    taken from : https://java2blog.com/object-of-type-int64-is-not-json-serializable/
    """
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        # if the object is a function, save it as a string
        if callable(obj):
            return str(obj)
        return super(NpEncoder, self).default(obj)