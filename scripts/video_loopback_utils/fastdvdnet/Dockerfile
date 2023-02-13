# Dockerfile from https://github.com/anibali/docker-pytorch

FROM nvidia/cuda:10.1-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.6
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.6.9 \
 && conda clean -ya
 
# CUDA 10.1-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    "pytorch==1.0.0" \
    "torchvision==0.2.1" \
    "cuda100==1.0" \
    "opencv==3.4.2" \
    "pycodestyle==2.5.0" \
    "pytest==5.4.1" \
    "scikit-image==0.16.2" \ 
 && conda clean -ya
 
RUN pip install tensorboardX==2.0
# Requirements
# RUN curl -sL https://raw.githubusercontent.com/ZurMaD/fastdvdnet/master/requirements.txt -o requirements.txt
# RUN pip install -r requirements.txt


# Set the default command to python3
CMD ["python3"]
