# Res2SE50

The data presented in this projects are publicly available at https://captain-whu.github.io/BED4RS/ and https://gcheng-nwpu.github.io/#Datasets.

This study was conducted under the Linux operating system using an Intel(R) Xeon(R) Platinum 8255C CPU, a GeForce RTX2080 GPU, and 40 GB of RAM. The experiments were implemented using PyTorch 1.11.0, Python 3.8 (ubuntu20.04) and Cuda 11.3. The experiments use migration learning techniques to initialize the residual module parameters in the Res2SE50 model using the ResNet50 model parameters obtained from training on the ImageNet29 dataset to improve the model convergence speed. Then, we fine-tune the Res2SE50 model using the AID and NWPU datasets to improve the performance of the model in the remote sensing image scene classification task. Before inputting into the network model, all images were resized to 224 × 224 and the batch size was set to 32. 
