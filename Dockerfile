# Use the Miniforge3 base image for ARM64 architecture, which provides a minimal conda installation
FROM --platform=linux/arm64/v8 condaforge/miniforge3

# Set the working directory in the container
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Clone the GitHub repository
RUN git clone https://github.com/uio-bmi/fdr_under_dependencies.git

# Set the working directory to the cloned repository
WORKDIR /app/fdr_under_dependencies

# Change to the specific branch
RUN git checkout users/mmamica/add-unit-tests

# Install git, git-lfs, and build-essential tools
RUN apt-get update && apt-get install -y git git-lfs build-essential

# Copy the data files to the container
COPY data/realworld_methyl_beta.h5 /app/fdr_under_dependencies/data/realworld_methyl_beta.h5

# Create a Conda environment
RUN conda create --name fdr_under_dependencies python=3.9

# Install R and Bioconductor limma package
RUN conda run -n fdr_under_dependencies conda install -c conda-forge r-base
RUN conda run -n fdr_under_dependencies conda install -c bioconda bioconductor-limma

# Install Python dependencies and the local package
RUN conda run -n fdr_under_dependencies pip install -r requirements.txt
RUN conda run -n fdr_under_dependencies pip install -r requirements_dev.txt
RUN conda run -n fdr_under_dependencies pip install .
