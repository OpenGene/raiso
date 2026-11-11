"""riasopy -- spatial omics data analysis (Rust core).

Thin shell package: the real implementation lives in the native extension
``riasopy.riasopy`` (the maturin build artifact). See ``riasopy.riasopy.__all__``
for ``__all__``.
"""

import os as _os
from pathlib import Path as _Path

# Plotting goes through plotters -> fontconfig. The wheel ships 22 DejaVu ttf files
# together with a fonts.conf (share/etc/fonts/fonts.conf, whose
# <dir prefix="relative"> points at share/fonts). Point FONTCONFIG_FILE there so
# rendering does not depend on the target machine's /usr/share/fonts (absent on many
# servers) nor on a conda fontconfig package. This has to happen before the extension
# module is imported: the bundled libfontconfig has the build machine's absolute
# config path compiled in, which does not exist on the target machine. setdefault
# keeps a value the user has already set.
_CONF = _Path(__file__).resolve().parent / "share" / "etc" / "fonts" / "fonts.conf"
if _CONF.is_file():
    _os.environ.setdefault("FONTCONFIG_FILE", str(_CONF))

del _os, _Path, _CONF

from .riasopy import *
from .riasopy import __all__

__version__ = "0.1.1"
