# Knowledge Graph Embedding

1. configuration

```
# Python version 3.6

# install CUDA
conda install -y cudatoolkit=10.0

# install cudnn libraries, to improve the performance of tensorflow
conda install cudnn=7.6

# Intall tensorflow GPU
pip install tensorflow-gpu

# Install AmpliGraph library
pip install ampligraph

# Required to visualize embeddings with tensorboard projector, comment out if not required!
pip install --user tensorboard

# Required to plot text on embedding clusters, comment out if not required!
pip install --user git+https://github.com/Phlya/adjustText
```

2. draw the graph
```
# first install graphviz for the system
windows: https://graphviz.org/download/, add the path to the system path
ubuntu: sudo apt-get install graphviz

# install graphviz package in the virtual environment 
pip install graphviz
```


3. dd 
