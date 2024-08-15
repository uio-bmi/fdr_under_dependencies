FDR under dependencies
===========

**FDR under dependencies** project aims to analyse the impact of variable dependencies on the performance of false discovery rate (FDR) procedures.
It provides a set of tools and pipelines for generating synthetic and semi-real-world data based on the real-world methylation dataset, as well as for conducting multiple hypotheses testing.

The pipelines are implemented using `Snakemake <https://snakemake.readthedocs.io/>`_, a robust workflow management system that facilitates easy scaling and parallelization of analyses.
This setup ensures that the project can handle large datasets efficiently while maintaining flexibility and reproducibility in the research process.

Installation guide
------------------

1. **Clone repository:**

   .. code-block:: bash

      git clone https://github.com/uio-bmi/fdr_hacking.git

2. **Create conda environment:**

   .. code-block:: bash

      conda create --name multiple_testing_analysis python=3.9
      conda activate multiple_testing_analysis

3. **Install R in your conda environment (it will be needed since we use rpy2) and needed R package:**

   .. code-block:: bash

      conda install -c conda-forge r-base
      conda install -c bioconda bioconductor-limma

4. **Restore large files from lfs:**

   .. code-block:: bash

      git lfs install
      git lfs pull

5. **Install requirements:**

   .. code-block:: bash

      pip install -r requirements.txt
      pip install -r requirements_dev.txt

6. **Install local project:**

   .. code-block:: bash

      pip install .

7. **You can verify installations by running the tests:**

   .. code-block:: bash

      pytest

8. **You can run simple analyses:**

   .. code-block:: bash

       snakemake -s pipelines/synthetic/Snakefile -d pipelines/synthetic --cores 4 --config workflow_config=../../config/dummy_synthetic_data.yaml

       snakemake -s pipelines/semi_real_world/Snakefile -d pipelines/semi_real_world --cores 4 --config workflow_config=../../config/dummy_semi_real_world_data.yaml
