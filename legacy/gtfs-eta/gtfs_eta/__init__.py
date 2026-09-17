"""
gtfs_eta — transitional alias namespace for this repo's top-level packages.

WHY THIS EXISTS
----------------
This repo ships four *physically* top-level packages — core/, eta_service/,
feature_engineering/, models/ — that a live docker stack (gtfs-rt-pipeline)
bind-mounts and imports directly (``import eta_service``, ``from
models.common.registry import get_registry``, etc.). Those directories
cannot move, be renamed, or change import behavior without breaking that
collector, so a physical restructure into a real ``gtfs_eta/`` package tree
is out of scope here — it happens later, at fork time.

The sibling ``databus`` Django project, meanwhile, wants to depend on this
repo as ``gtfs_eta.*`` (a uv editable path dependency). This package bridges
that gap in the meantime: it makes ``gtfs_eta.<pkg>[.<submodule>...]``
resolve to the *exact same module object* as ``<pkg>[.<submodule>...]``, for
every submodule of core/eta_service/feature_engineering/models, lazily and
on demand.

MECHANISM
---------
A ``sys.meta_path`` finder (``_AliasFinder`` below) intercepts any import
whose top-level segment after "gtfs_eta." names one of the four aliased
packages, imports the corresponding real dotted name via
``importlib.import_module`` (which is itself cached/idempotent — if the
real module is already imported, this is a no-op lookup), and hands that
*same* module object back to Python's import machinery as the result. No
new module is ever executed under the "gtfs_eta.*" name.

This was chosen over the simpler alternative of extending this package's
``__path__`` to include the repo root (so plain package-finding would
"discover" core/, eta_service/, etc. as gtfs_eta's own subpackages):
``__path__`` extension makes Python's import system execute each
submodule's source a *second* time under the "gtfs_eta.*" name, producing a
second, independent module object with its own copy of any module-level
singleton — e.g. models/common/registry.py's ``_registry`` global would
exist twice, once per import spelling, silently defeating the model
registry's singleton contract. The sys.modules aliasing here guarantees
``models.common.registry`` and ``gtfs_eta.models.common.registry`` are
literally the same module (verified: same ``sys.modules`` entry, same
``_registry`` instance, regardless of which spelling is imported first).

The only observable side effect is cosmetic: a module reached via the
"gtfs_eta.*" spelling has its ``__spec__.name`` reflect that spelling
(Python's import machinery always stamps ``module.__spec__ = spec``,
unconditionally) — ``__name__`` and ``__package__`` are left untouched.
Nothing in this codebase inspects ``__spec__.name``, and the live collector
never imports ``gtfs_eta`` at all, so it never observes this finder.

Delete this file (and the meta path finder it installs) once core/,
eta_service/, feature_engineering/, and models/ physically move under
gtfs_eta/ at fork time — at that point "gtfs_eta.*" becomes the only
spelling and this indirection is no longer needed.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import sys

# The four packages this alias namespace re-exports. Keep in sync with
# [tool.setuptools.packages.find] in pyproject.toml.
_ALIASED_TOP_LEVEL_PACKAGES = ("core", "eta_service", "feature_engineering", "models")


class _AliasFinder:
    """Redirects ``gtfs_eta.<pkg>...`` imports onto the real top-level module.

    Installed on ``sys.meta_path`` (ahead of the default path-based finder)
    so it gets first refusal on every import. Only claims names under the
    four aliased packages; everything else (e.g. ``gtfs_eta.seed_baseline_model``,
    a real file living in this directory) falls through to normal resolution.
    """

    def find_spec(self, fullname: str, path, target=None):
        if not self._matches(fullname):
            return None
        real_name = fullname[len(__name__) + 1:]
        real_module = importlib.import_module(real_name)
        spec = importlib.machinery.ModuleSpec(
            fullname, self, is_package=hasattr(real_module, "__path__")
        )
        # Stash the already-imported real module; create_module hands it
        # straight back so no second copy of the source is ever executed.
        spec.loader_state = real_module
        return spec

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        return spec.loader_state

    def exec_module(self, module) -> None:
        # module is the real module object, already fully executed — nothing
        # left to do.
        pass

    @staticmethod
    def _matches(fullname: str) -> bool:
        prefix = f"{__name__}."
        if not fullname.startswith(prefix):
            return False
        top_segment = fullname[len(prefix):].split(".", 1)[0]
        return top_segment in _ALIASED_TOP_LEVEL_PACKAGES


sys.meta_path.insert(0, _AliasFinder())
