# hypergraph-learning-based-discriminative-band-selection
For hyperspectral images (HSIs), it is a challenging task to select discriminative bands due to the lack of labeled samples and complex noise. In this article, we present a novel local-view-assisted discriminative band selection method with hypergraph autolearning (LvaHAl) to solve these problems from both local and global perspectives. Specifically, the whole band space is first randomly divided into several subspaces (LVs) of different dimensions, where each LV denotes a set of lowdimensional representations of training samples consisting of bands associated with it. Then, for different LVs, a robust hinge loss function for isolated pixels regularized by the row-sparsity is adopted to measure the importance of the corresponding bands. In order to simultaneously reduce the bias of LVs and encode the complementary information between them, samples from all LVs are further projected into the label space. Subsequently, a hypergraph model that automatically learns the hyperedge weights is presented. In this way, the local manifold structure of these projections can be preserved, ensuring that samples of the same class have a small distance. Finally, a consensus matrix is used to integrate the importance of bands corresponding to different LVs, resulting in the optimal selection of expected bands from a global perspective. The classification experiments on three HSI data sets show that our method is competitive with other comparison methods.


Links to the test dataset: https://pan.baidu.com/s/1Ny5XRR0AZqCBTd_qZrY6-A
Fetch code: 2cne


If this code is useful for you, please cite the following related work


@article{wei2020local-view-assisted,


title={Local-View-Assisted Discriminative Band Selection With Hypergraph Autolearning for Hyperspectral Image Classification},


author={Wei, Xiaohui and Cai, Lijun and Liao, Bo and Lu, Ting},


journal={IEEE Transactions on Geoscience and Remote Sensing},


volume={58},


number={3},


pages={2042--2055},


year={2020}}
