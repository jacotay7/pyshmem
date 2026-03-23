from __future__ import annotations

import uuid

import pytest

import pyshare


@pytest.fixture
def shm_name():
    name = f"pyshare_{uuid.uuid4().hex}"
    yield name
    pyshare.unlink(name)
