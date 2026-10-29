import pytest

from biocimpute import _kernels


@pytest.fixture(params=_kernels.BACKENDS)
def backend(request, monkeypatch):
    """Run a test once per kernel backend."""
    try:
        module = _kernels._load(request.param)
    except ImportError as e:
        pytest.skip(str(e))
    monkeypatch.setattr(_kernels, "_backend", module)
    return module
