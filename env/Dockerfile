FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel
RUN apt-get update && apt-get install -y git
# Install dependency.
RUN pip install -U pip
RUN pip install numpy pandas seaborn tensorboard
RUN pip install torch_geometric==2.5.1
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
RUN conda install lightning -c conda-forge
RUN pip install lightning[extra]
RUN pip install matplotlib==3.8.3
RUN pip install biopandas==0.5.1
RUN pip install biopython==1.83
RUN pip install transformers==4.40.2
RUN pip install sentencepiece==0.2.0
RUN pip install torchsummary==1.5.1
RUN pip install scipy==1.12.0
RUN pip install torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
RUN pip install git+https://github.com/yusuf1759/prodigy-cryst.git
RUN pip install pinder[all]
RUN pip install fair-esm
