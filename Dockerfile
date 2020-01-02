FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates && \
     rm -rf /var/lib/apt/lists/*
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing\
      pytorch torchvision cudatoolkit=10.0 -c pytorch&& \
     #/opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN pip install nltk transformers python-docx wordcloud pdfminer scikit-learn mxnet gluonnlp tensorboardX torchtext sentencepiece



#/redis-5.0.0/src/redis-server
#CMD [ "cd", "./redis-5.0.0/src/" ]
#CMD ["redis-server"]
