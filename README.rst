FDR Hacking
===========

Installation guide
------------------

1. **Clone repository:**

   .. code-block:: bash

      git clone https://github.com/uio-bmi/fdr_hacking.git

2. **Create conda environment:**

   .. code-block:: bash

      conda create --name multiple_testing_analysis python=3.9
      conda activate multiple_testing_analysis

3. **Install R in your conda environment (it will be needed since we use rpy2) and needed R package**

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
