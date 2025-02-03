import pathlib

__version__ = "0.0.1"

here = pathlib.Path(__file__).parent.resolve()

def get_version(rel_path="__init__.py"):
    init_content = (here / rel_path).read_text(encoding='utf-8')
    for line in init_content.split('\n'):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")
