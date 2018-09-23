# Hyperspectral-Image-Super-Resolution-Benchmark
A list of hyperspectral image super-solution resources collected by Junjun Jiang. If you find that important resources are not included, please feel free to contact me.

According to whether or not to use auxiliary images (PAN image/RGB image/multispectral images), hyperspectral image super-solution techniques can be divided into two classes: hyperspectral image super-solution (fusion) and single hyperspectral image super-solution. The former could be roughly categorized as follows: 1) Bayesian based approaches, 2) Tensor based approaches, 3) Matrix factorization based approaches, and 4) Deep Learning based approaches. 

#### Pioneer work:
Unmixing based multisensor multiresolution image fusion, TGRS1999, B. Zhukov et al.

Application of the stochastic mixing model to hyperspectral resolution enhancement, TGRS2004, M. T. Eismann et al.

Resolution enhancement of hyperspectral imagery using maximum a posteriori estimation with a stochastic mixing model, Ph.D. dissertation, 2004, M. T. Eismann et al.

MAP estimation for hyperspectral image resolution enhancement using an auxiliary sensor, TIP2004, R. C. Hardie et al.

Hyperspectral resolution enhancement using high-resolution multispectral imagery with arbitrary response functions, TGRS2005, M. T. Eismann et al.

#### Technique Review:
Hyperspectral pansharpening: a review. GRSM2015, L. Loncan et al.
[[PDF](http://wei.perso.enseeiht.fr/papers/HyperPAN_review_2015.pdf) 
[[Code](http://wei.perso.enseeiht.fr/data/Results_GRSM_Qi%20WEI.zip)

Hyperspectral and multispectral data fusion: A comparative review of the recent literature, GRSM2017, N. Yokoya,et al.
[[PDF](http://naotoyokoya.com/assets/pdf/NYokoyaGRSM2017.pdf)
[[Code](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip)


#### Hyperspectral image super-solution (fusion)
###### Bayesian based approaches:
- Blind Image Fusion for Hyperspectral Imaging with the Directional Total Variation, Inverse Problems, 2018, Leon Bungert et al.
[[PDF](https://arxiv.org/abs/1710.05705)
[[Code](https://github.com/mehrhardt/blind_remote_sensing)

- Bayesian sparse representation for hyperspectral image super resolution, CVPR2015, N. Akhtar et al.
[[PDF]( https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Akhtar_Bayesian_Sparse_Representation_2015_CVPR_paper.pdf
[[Code](http://openremotesensing.net/wp-content/uploads/2016/12/Supplementary.zip)

- Hysure: A convex formulation for hyperspectral image superresolution via subspace-based regularization, TGRS2015, M. Simoes et al.
[[PDF](https://arxiv.org/abs/1411.4005)
[[Code](https://github.com/alfaiate/HySure)

- Hyperspectral and multispectral image fusion based on a sparse representation, TGRS2015, Q. Wei et al.
[[PDF](https://arxiv.org/pdf/1409.5729)
[[Code](http://wei.perso.enseeiht.fr/demo/SparseFusion_2014-12-03.zip)

- Bayesian fusion of multi-band images, Jstar2015, W. Qi et al.
[[PDF](http://wei.perso.enseeiht.fr/papers/WEI_JSTSP_final.pdf)
[[Code](http://wei.perso.enseeiht.fr/demo/MCMCFusion.7z)

- Noise-resistant wavelet-based Bayesian fusion of multispectral and hyperspectral images, TGRS2009, Y. Zhang et al.
[[PDF](https://ieeexplore.ieee.org/iel5/36/4358825/04967929.pdf)

- Weighted Low-rank Tensor Recovery for Hyperspectral Image Restoration, arXiv2018, Yi Chang et al.
[[PDF](https://arxiv.org/pdf/1709.00192.pdf)

###### Tensor based approaches:
- Hyperspectral image superresolution via non-local sparse tensor factorization, CVPR2017, R. Dian et al.
[[PDF](https://sites.google.com/site/leyuanfang/cvpr-17)

- Spatial–Spectral-Graph-Regularized Low-Rank Tensor Decomposition for Multispectral and Hyperspectral Image Fusion, Jstars2018, K. Zhang et al.
[[PDF](https://www.researchgate.net/publication/322559599_Spatial-Spectral-Graph-Regularized_Low-Rank_Tensor_Decomposition_for_Multispectral_and_Hyperspectral_Image_Fusion)

- Fusing Hyperspectral and Multispectral Images via Coupled Sparse Tensor Factorization, TIP2108, S. Li et al.
[[PDF](https://drive.google.com/open?id=1ZJQB1RhjRO9JNTBVaNknk1DXjiWfg6Gd)
[[Code](https://drive.google.com/open?id=12eleEjv7wKQxFCBUcIGkEl-wiUiJxwTv)

- Hyperspectral Super-Resolution: A Coupled Tensor Factorization Approach, arXiv2018, Charilaos I. Kanatsoulis et al.
[[PDF](https://arxiv.org/pdf/1804.05307.pdf)

###### Matrix factorization based approaches:
- High-resolution hyperspectral imaging via matrix factorization, CVPR2011, R. Kawakami et al.
[[PDF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.3532&rep=rep1&type=pdf)
[[Code](http://www.nae-lab.org/~rei/research/hh/index.html)

- Coupled nonnegative matrix factorization unmixing for hyperspectral and multispectral data fusion, TGRS2012, N. Yokoya et al.
[[PDF](http://naotoyokoya.com/assets/pdf/NYokoyaTGRS2012.pdf)
[[Code](http://naotoyokoya.com/assets/zip/CNMF_MATLAB.zip)

- Sparse spatio-spectral representation for hyperspectral image super-resolution, ECCV2014, N. Akhtar et al.
[[PDF](http://openremotesensing.net/wp-content/uploads/2016/12/ECCV2014_Naveed.pdf)
[[Code](http://openremotesensing.net/wp-content/uploads/2016/12/HSISuperRes.zip)

- Hyper-sharpening: A first approach on SIM-GA data, Jstars2015, M. Selva et al.

- Hyperspectral super-resolution by coupled spectral unmixing, ICCV2015, C Lanaras.
https://github.com/lanha/SupResPALM
[[PDF](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lanaras_Hyperspectral_Super-Resolution_by_ICCV_2015_paper.pdf)

- RGB-guided hyperspectral image upsampling, CVPR2015, H. Kwon et al.
[[PDF](https://pdfs.semanticscholar.org/2dfb/a20d04832e6ee7076c695f7bb99dcf1a3f02.pdf)
[[Code](https://sites.google.com/site/rgbhsupsampling/)

- Multiband image fusion based on spectral unmixing, TGRS2016, Q. Wei et al.
[[PDF](https://arxiv.org/abs/1603.08720) 
[[Code](https://github.com/qw245/FUMI)

- Hyperspectral image super-resolution via non-negative structured sparse representation, TIP2016, W. Dong, et al.
http://see.xidian.edu.cn/faculty/wsdong/Code_release/NSSR_HSI_SR.rar
[[PDF](http://see.xidian.edu.cn/faculty/wsdong/Papers/Journal/NSSR_HSI_TIP16.pdf

- Hyperspectral super-resolution of locally low rank images from complementary multisource data, TIP2016, M. A. Veganzones et al.
[[PDF](https://hal.archives-ouvertes.fr/hal-00960076/document)

- Multispectral and hyperspectral image fusion based on group spectral embedding and low-rank factorization, TGRS2017, K. Zhang et al.

- Hyperspectral Image Super-Resolution Based on Spatial and Spectral Correlation Fusion, TRGS2018, C. Yi et al.

- Super-Resolution for Hyperspectral and Multispectral Image Fusion Accounting for Seasonal Spectral Variability, arXiv2018, Ricardo Augusto Borsoi et al.
[[PDF](https://arxiv.org/abs/1808.10072)

- Self-Similarity Constrained Sparse Representation for Hyperspectral Image Super-Resolution, TIP2108, X. Han et al.
[[PDF](https://www.researchgate.net/publication/257879745_Hyperspectral_imagery_super-resolution_by_sparse_representation_and_spectral_regularization)

- Exploiting Clustering Manifold Structure for Hyperspectral Imagery Super-Resolution, TIP2018, L. Zhang et al.
https://ieeexplore.ieee.org/document/8424415/
[[PDF](https://sites.google.com/site/leizhanghyperspectral/publications)

- Hyperspectral Image Super-Resolution With a Mosaic RGB Image, TIP2018, Y. Fu et al.
[[PDF](https://ieeexplore.ieee.org/document/8410569/)

###### Deep Learning based approaches:
- Deep Residual Convolutional Neural Network for Hyperspectral Image Super-Resolution, ICIG2017, C. Wang et al.

SSF-CNN: Spatial and Spectral Fusion with CNN for Hyperspectral Image Super-Resolution, ICIP2018, X. Han et al.

- Deep Hyperspectral Image Sharpening, TNNLS2018, R. Dian et al.
[[PDF](https://drive.google.com/open?id=1FIyVL9c8jlDY3heEZ57nGvpSDZc0mkeT)
[[Code](https://drive.google.com/open?id=19xYNnCht1-_nh4pys6Fw7z0mVQqRha8k)


- HSI-DeNet: Hyperspectral Image Restoration via Convolutional Neural Network, TGRS2018, Y. Chang et al.
[[Code](http://www.escience.cn/people/changyi/index.html)

- Unsupervised Sparse Dirichlet-Net for Hyperspectral Image Super-Resolution, CVPR2018, Y. Qu et al.
[[PDF](https://arxiv.org/abs/1804.05042)

#### Single Hyperspectral Image Super-Resolution

- Super-resolution reconstruction of hyperspectral images, TIP2005, T. Akgun et al.
Paper: https://ieeexplore.ieee.org/document/1518950/

- Enhanced self-training superresolution mapping technique for hyperspectral imagery, GRSL2011, F. A. Mianji et al.

- A super-resolution reconstruction algorithm for hyperspectral images. Signal Process. 2012, H. Zhang et al.

Super-resolution hyperspectral imaging with unknown blurring by low-rank and group-sparse modeling, ICIP2014, H. Huang et al.

- Super-resolution mapping via multi-dictionary based sparse representation, ICASSP2016, H. Huang et al.

- Super-resolution: An efficient method to improve spatial resolution of hyperspectral images, IGARSS2016, A. Villa, J. Chanussot et al.

- Hyperspectral image super resolution reconstruction with a joint spectral-spatial sub-pixel mapping model, IGARSS2016, X. Xu et al.

- Hyperspectral image super-resolution by spectral mixture analysis and spatial–spectral group sparsity, GRSL2016, J. Li et al.

- Super-resolution reconstruction of hyperspectral images via low rank tensor modeling and total variation regularization, IGARSS2016, S. He et al.
https://arxiv.org/abs/1601.06243

- Hyperspectral image super-resolution by spectral difference learning and spatial error correction, GRSL2017, J. Hu et al.

- Hyperspectral image superresolution by transfer learning, Jstars2017, Y. Yuan et al.

- A MAP-Based Approach for Hyperspectral Imagery Super-Resolution, TIP2018, Hasan Irmak et al.

