import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATA_DIR", os.path.join(_tmp, "data"))
os.environ.setdefault("CONFIG_DIR", os.path.join(_tmp, "config"))
os.makedirs(os.environ["DATA_DIR"], exist_ok=True)
