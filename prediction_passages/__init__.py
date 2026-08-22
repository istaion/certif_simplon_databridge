import sys
from pathlib import Path

# Ajoute prediction_passages/ au sys.path pour que les imports
# "from src.xxx import ..." fonctionnent quand le package est importé
# depuis la racine du projet (ex: from prediction_passages.main import ...).
_pkg_dir = str(Path(__file__).parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
