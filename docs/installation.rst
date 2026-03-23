Installation
============

Base install
------------

Install the core package with:

.. code-block:: bash

   pip install pyshare

For local development from a checkout:

.. code-block:: bash

   pip install -e .

Optional dependencies
---------------------

Testing tools:

.. code-block:: bash

   pip install -e .[test]

CUDA-backed GPU support:

.. code-block:: bash

   pip install -e .[gpu]

Documentation build dependencies:

.. code-block:: bash

   pip install -e .[docs]

Requirements
------------

- Python 3.9 or newer
- NumPy 1.26 or newer
- PyTorch with CUDA support if you want GPU streams

To confirm GPU availability inside your environment:

.. code-block:: python

   import pyshare
   print(pyshare.gpu_available())