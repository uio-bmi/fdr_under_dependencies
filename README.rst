FDR under Dependencies
===========

In this work, we demonstrate that in datasets with highly correlated features, as in omics, FDR correction methods like Benjamini-Hochberg can within their formal guarantees in some cases report very high numbers of false positives.
This can lead to thousands of falsely reported findings, even if all null hypotheses are true.
For more information, see the preprint here: URL to be updated when available.

The project repo contains the source code we used to produce the results in the manuscript (URL to be updated).
Below, we provide instructions on how to reproduce the results using conda environment or supplied docker containers.

Installation guide (via Conda)
------------------

1. **Clone repository:**

   .. code-block:: bash

      git clone https://github.com/uio-bmi/fdr_under_dependencies.git

2. **Create conda environment:**

   .. code-block:: bash

      conda create --name fdr_under_dependencies python=3.9
      conda activate fdr_under_dependencies

3. **Install R in your conda environment (it will be crucial since we use rpy2) and needed R package:**

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

7. **You can verify installations by running the tests and simple analyses:**

   .. code-block:: bash

      pytest

      snakemake -s pipelines/synthetic/Snakefile -d pipelines/synthetic --cores 4 --config workflow_config=../../config/dummy_synthetic_data.yaml

      snakemake -s pipelines/semi_real_world/Snakefile -d pipelines/semi_real_world --cores 4 --config workflow_config=../../config/dummy_semi_real_world_data.yaml

Installation guide (via Docker)
------------------
1. **Pull the Docker image:**

   If you are using an ARM machine, we suggest using the following image:

   .. code-block:: bash

      docker pull mmamica/fdr_under_dependencies:arm

   And if you are using an AMD machine, we suggest:

   .. code-block:: bash

      docker pull mmamica/fdr_under_dependencies:amd

2. **Run the Docker container:**

   For ARM machines:

   .. code-block:: bash

      docker run -it mmamica/fdr_under_dependencies:arm

   And for AMD:

   .. code-block:: bash

      docker run -it mmamica/fdr_under_dependencies:amd

3. **Activate the conda environment:**

   .. code-block:: bash

      conda activate fdr_under_dependencies

4. **You can verify installations by running the tests and simple analyses:**

   .. code-block:: bash

      pytest

      snakemake -s pipelines/synthetic/Snakefile -d pipelines/synthetic --cores 4 --config workflow_config=../../config/dummy_synthetic_data.yaml

      snakemake -s pipelines/semi_real_world/Snakefile -d pipelines/semi_real_world --cores 4 --config workflow_config=../../config/dummy_semi_real_world_data.yaml

Replicating the results
------------------
In order to replicate the results, you need to run the following commands:

.. code-block:: bash

      snakemake -s pipelines/synthetic/Snakefile -d pipelines/synthetic --cores 4 --config workflow_config=../../config/synthetic_data.yaml

      snakemake -s pipelines/semi_real_world/Snakefile -d pipelines/semi_real_world --cores 4 --config workflow_config=../../config/semi_real_world_data.yaml

Results will be stored in the `results` directory.
Remember that the analyses are computationally expensive and can take a long time to complete.
We suggest running the analyses on a machine with a high number of cores and a large amount of RAM.
