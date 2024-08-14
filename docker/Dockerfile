# Use an x86-64 based image
FROM --platform=linux/amd64 continuumio/anaconda3

# Set the working directory in the container
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Clone the GitHub repository
RUN git clone https://github.com/uio-bmi/fdr_hacking.git

# Set the working directory to the cloned repository
WORKDIR /app/fdr_hacking

# Change to the specific branch
RUN git checkout users/mmamica/add-unit-tests

# Install git, git-lfs, and build-essential tools
RUN apt-get update && apt-get install -y git git-lfs build-essential

# Pull large files
RUN git lfs install
RUN git lfs pull

# Create a Conda environment
RUN conda create --name multiple_testing_analysis python=3.9

# Install R and Bioconductor limma package
RUN conda run -n multiple_testing_analysis conda install -c conda-forge r-base
RUN conda run -n multiple_testing_analysis conda install -c bioconda bioconductor-limma

# Install Python dependencies and the local package
RUN conda run -n multiple_testing_analysis pip install -r requirements.txt
RUN conda run -n multiple_testing_analysis pip install -r requirements_dev.txt
RUN conda run -n multiple_testing_analysis pip install .
