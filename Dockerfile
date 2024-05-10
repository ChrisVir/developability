FROM mambaorg/micromamba

ARG MAMBA_DOCKERFILE_ACTIVATE=1

USER root

WORKDIR /app

RUN apt-get update \
    && apt-get -y install nano sudo wget zip procps \ 
    && wget https://github.com/Electrostatics/apbs/releases/download/v3.4.1/APBS-3.4.1.Linux.zip \
    && unzip APBS-3.4.1.Linux.zip \
    && rm APBS-3.4.1.Linux.zip \
    && apt-get clean

RUN  micromamba config append channels conda-forge \
    && micromamba config append channels bioconda \
    && micromamba install python=3.11 biopython abnumber openmm ambertools ipython biopandas faiss "biobb_amber>=4.1.0" pdbfixer click pip seaborn sarge pdb2pqr jinja2 tqdm  pytorch -y \
    && micromamba install electrostatics::nanoshaper -y \
    && micromamba clean -a -y \
    && pip install ImmuneBuilder

ENV PATH="$PATH:/app/APBS-3.4.1.Linux/bin:/app/APBS-3.4.1.Linux/share/apbs/tools/bin" 

COPY . /app/developability/

RUN cd /app/developability \
 && pip install -e. 

#USER app_user

