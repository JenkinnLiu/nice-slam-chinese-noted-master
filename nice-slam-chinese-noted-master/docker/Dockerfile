FROM continuumio/anaconda3

# Prepare and empty machine for building.
RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
      g++ \
      zlib1g-dev \
      libopenexr-dev \
      ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# conda env
COPY ./ /root
WORKDIR /root
RUN conda env create -f environment.yaml && rm environment.yaml 
