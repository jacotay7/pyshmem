from __future__ import annotations

import uuid

import pytest

import pyshmem


@pytest.fixture
def shm_name():
    name = f"pyshmem_{uuid.uuid4().hex}"
    yield name
    pyshmem.unlink(name)
