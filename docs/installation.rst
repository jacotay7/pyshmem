Installation
============

Requirements
------------

- Python 3.9 or newer
- NumPy 1.26 or newer
- ``portalocker`` 3.1 or newer (installed automatically)

For GPU streams:

- PyTorch 2.2 or newer, built with CUDA support
- A CUDA-capable GPU

Base install
------------

Install the core package from PyPI:

.. code-block:: bash

   pip install pyshmem

This provides CPU shared-memory streams for NumPy arrays.  No PyTorch or CUDA
dependency is required.

Optional extras
---------------

GPU support:

.. code-block:: bash

   pip install pyshmem[gpu]

This pulls in ``torch>=2.2``.  A separate CUDA toolkit installation is not
required if you use a PyTorch wheel that bundles CUDA.

Testing tools:

.. code-block:: bash

   pip install pyshmem[test]

Documentation build dependencies:

.. code-block:: bash

   pip install pyshmem[docs]

Development install
-------------------

For a local checkout with all test dependencies:

.. code-block:: bash

   git clone https://github.com/jacotay7/pyshmem.git
   cd pyshmem
   pip install -e .[test]

For GPU development:

.. code-block:: bash

   pip install -e .[test,gpu]

Verifying the install
---------------------

Check that the package is importable and report GPU availability:

.. code-block:: python

   import pyshmem
   print(pyshmem.__version__)
   print("GPU available:", pyshmem.gpu_available())

The CLI should also be available after installation:

.. code-block:: bash

   pyshmem --help
