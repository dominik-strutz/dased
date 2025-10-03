.. DASED documentation master file, created by
   sphinx-quickstart on Thu Jul  3 08:37:30 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DASED documentation
===================

Welcome to the DASED documentation! This site provides an overview of the project, its modules, and usage examples.

.. note::

   This documentation is a work in progress. There is no comprehensive test suite yet. Everything that goes beyond the applications presented in the tutorials or the code for the accompanying research paper is not guaranteed to work. Contributions are welcome!


Installation
------------

To install the package, simply run

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install git+https://github.com/dominik-strutz/dased.git

   .. tab-item:: uv

      .. code-block:: bash

         uv add git+https://github.com/dominik-strutz/dased.git

   .. tab-item:: conda

      .. code-block:: bash

         conda install git+https://github.com/dominik-strutz/dased.git


.. note::
    The package is still in heavy development and can change rapidly. If you want to use it, its recommended to fix the version by running

    .. code-block:: console

        pip install git+https://github.com/dominik-strutz/dased@<version>

    where `<version>` is the version you want to use (e.g. a commit hash or a tag).


Usage
=====

See the :doc:`tutorials` and :doc:`api_reference` sections for more information.



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   api_reference
   tutorials
