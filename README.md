# Hyperspectral-Image-Super-Resolution-Benchmark
A list of hyperspectral image super-resolution resources collected by [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun). If you find that important resources are not included, please feel free to contact me.

Hyperspectral image super-resolution is a kind of technique that can generate a high spatial and high spectral resolution image from one of the following obsearved data (1) low-resolution multispectral image, e.g., RGB image, (2) low-resolution hyperspectral image, or (3) high-resolution multispectral image and low-resolution hyperspectral image. According to kind of obsearved data, hyperspectral image super-resolution techniques can be divided into two classes: joint spatial and spectral super-resolution, i.e., spatiospectral super-resolution (SSSR), single hyperspectral image super-resolution (SHSR), and multispectral image and hyperspectral image (MHF).

================================================================================
### 0. Pioneer Work and Technique Review
- Unmixing based multisensor multiresolution image fusion, TGRS1999, B. Zhukov et al.

- Application of the stochastic mixing model to hyperspectral resolution enhancement, TGRS2004, M. T. Eismann et al.

- Resolution enhancement of hyperspectral imagery using maximum a posteriori estimation with a stochastic mixing model, Ph.D. dissertation, 2004, M. T. Eismann et al.

- MAP estimation for hyperspectral image resolution enhancement using an auxiliary sensor, TIP2004, R. C. Hardie et al.

- Hyperspectral resolution enhancement using high-resolution multispectral imagery with arbitrary response functions, TGRS2005, M. T. Eismann et al.

- Hyperspectral pansharpening: a review. GRSM2015, L. Loncan et al.
[[PDF](http://wei.perso.enseeiht.fr/papers/HyperPAN_review_2015.pdf)] 
[[Code](http://wei.perso.enseeiht.fr/data/Results_GRSM_Qi%20WEI.zip)]

- Hyperspectral and multispectral data fusion: A comparative review of the recent literature, GRSM2017, N. Yokoya,et al.
[[PDF](http://naotoyokoya.com/assets/pdf/NYokoyaGRSM2017.pdf)]
[[Code](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip)]

================================================================================

### 1. SpatioSpectral Super-Resolution (SSSR)
- Spatial and spectral joint super-resolution using convolutional neural network, TGRS 2020, S. Mei et al. [[PDF]https://ieeexplore.ieee.org/abstract/document/8974222]
- Multi-task Interaction learning for Spatiospectral Image Super-Resolution, Q. Ma et al. submitted to IEEE TIP, in peer review.
- Deep Unfolding Network for Spatiospectral Image Super-Resolution, Q. Ma et al. submitted to IEEE TCI, minor revision. [[Code]https://github.com/junjun-jiang/US3RN]


### 2. Single Hyperspectral Image Super-Resolution (SHSR)

- Super-resolution reconstruction of hyperspectral images, TIP2005, T. Akgun et al.

- Enhanced self-training superresolution mapping technique for hyperspectral imagery, GRSL2011, F. A. Mianji et al.

- A super-resolution reconstruction algorithm for hyperspectral images. Signal Process. 2012, H. Zhang et al.

- Super-resolution hyperspectral imaging with unknown blurring by low-rank and group-sparse modeling, ICIP2014, H. Huang et al.

- Super-resolution mapping via multi-dictionary based sparse representation, ICASSP2016, H. Huang et al.

- Super-resolution: An efficient method to improve spatial resolution of hyperspectral images, IGARSS2016, A. Villa, J. Chanussot et al.

- Hyperspectral image super resolution reconstruction with a joint spectral-spatial sub-pixel mapping model, IGARSS2016, X. Xu et al.

- Hyperspectral image super-resolution by spectral mixture analysis and spatial–spectral group sparsity, GRSL2016, J. Li et al.

- Super-resolution reconstruction of hyperspectral images via low rank tensor modeling and total variation regularization, IGARSS2016, S. He et al.
[[PDF](https://arxiv.org/abs/1601.06243)]

- Hyperspectral image super-resolution by spectral difference learning and spatial error correction, GRSL2017, J. Hu et al.

- Super-Resolution for Remote Sensing Images via Local–Global Combined Network, GRSL2017, J. Hu et al.

- Hyperspectral image superresolution by transfer learning, Jstars2017, Y. Yuan et al. [[PDF](http://ieeexplore.ieee.org/iel7/4609443/4609444/07855724.pdf)]

- Hyperspectral image super-resolution using deep convolutional neural network, Neurocomputing, 2017, Sen Lei et al. [[PDF](https://www.researchgate.net/publication/317024713_Hyperspectral_image_super-resolution_using_deep_convolutional_neural_network)]

- Hyperspectral image super-resolution via nonlocal low-rank tensor approximation and total variation regularization, Remote Sensing, 2017, Yao Wang et al. [[PDF](https://www.mdpi.com/2072-4292/9/12/1286/htm)]

- Hyperspectral Image Spatial Super-Resolution via 3D Full Convolutional Neural Network, Remote Sensing, 2017, Saohui Mei et al. [[PDF](https://www.mdpi.com/2072-4292/9/11/1139)]
[[Code](https://github.com/MeiShaohui/Hyperspectral-Image-Spatial-Super-Resolution-via-3D-Full-Convolutional-Neural-Network)]

- A MAP-Based Approach for Hyperspectral Imagery Super-Resolution, TIP2018, Hasan Irmak et al.

- Single Hyperspectral Image Super-resolution with Grouped Deep Recursive Residual Network, BigMM2018, Yong Li et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8499097)]
[[Code](https://github.com/Liyong8490/HSI-SR-GDRRN)]

- Hyperspectral image super-resolution with spectral–spatial network, IJRS2018, Jinrang Jia et al. [[PDF](https://www.tandfonline.com/doi/full/10.1080/01431161.2018.1471546)]

- Separable-spectral convolution and inception network for hyperspectral image super-resolution, IJMLC 2019, Ke Zheng et al.

- Hyperspectral Image Super-Resolution Using Deep Feature Matrix Factorization, IEEE TGRS 2019, Weiying Xie et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8678710)]

- Deep Hyperspectral Prior Single-Image Denoising, Inpainting, Super-Resolution, ICCVW2019, Oleksii Sidorov  et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8678710)]

- Spatial-Spectral Residual Network for Hyperspectral Image Super-Resolution, arXiv2020, Qi Wang et al. [[PDF](https://arxiv.org/pdf/2001.04609.pdf)]

- CNN-Based Super-Resolution of Hyperspectral Images, IEEE TGRS 2020, P. V. Arun et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9018371)]

- Hyperspectral Image Super-Resolution via Intrafusion Network, IEEE TGRS 2020, Jing Hu et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9057496)]

- Mixed 2D/3D Convolutional Network for Hyperspectral Image Super-Resolution, Remote Sensing 2020, Qiang Li et al. [[Code](https://github.com/qianngli/MCNet)][[Pdf](https://www.mdpi.com/2072-4292/12/10/1660)]

- Hyperspectral Image Super-Resolution by Band Attention Through Adversarial Learning, IEEE TGRS 2020, Jiaojiao Li et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/8960413)] 

- Learning Spatial-Spectral Prior for Super-Resolution of Hyperspectral Imagery, IEEE TCI 2020, Junjun Jiang et al. [[Code](https://github.com/junjun-jiang/SSPSR)][[Pdf](https://arxiv.org/abs/2005.08752)] **This is our method and achieves state-of-the-art performance for Single Hyperspectral Image Super-Resolution (SHSR) task**

- Bidirectional 3D Quasi-Recurrent Neural Networkfor Hyperspectral Image Super-Resolution, IEEE JStars 2021, Ying Fu et al. [[Web](https://ying-fu.github.io/)][[Pdf](https://ieeexplore.ieee.org/document/9351612)] 

- Hyperspectral Image Super-Resolution Using Spectrum and Feature Context, IEEE TIM 2021, Qi Wang et al. [[Web](https://crabwq.github.io/)][[Pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9264655)] 

- Hyperspectral Image Super-Resolution with Spectral Mixup and Heterogeneous Datasets, arXiv2021, Ke Li et al. [[Pdf](https://arxiv.org/abs/2101.07589)] 

- A Spectral Grouping and Attention-Driven Residual Dense Network for Hyperspectral Image Super-Resolution, IEEE TGRS 2021, Denghong Liu et al. [[Web](http://qqyuan.users.sgg.whu.edu.cn/)][[Pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9329109)] 

- Spatial-Spectral Feedback Network for Super-Resolution of Hyperspectral Imagery, arXiv 2021, Enhai Liu et al. [[Web](https://scholar.google.com.hk/citations?user=sgWhbbAAAAAJ&hl=zh-CN)][[Pdf](https://arxiv.org/pdf/2103.04354.pdf)] 

- Exploring the Relationship Between 2D/3D Convolution for Hyperspectral Image Super-Resolution, IEEE TGRS 2021, Qi Wang et al. [[Web](https://crabwq.github.io/)][[Pdf](https://ieeexplore.ieee.org/abstract/document/9334383)] 

- Hyperspectral Image Super-Resolution via Recurrent Feedback Embedding and Spatial-Spectral Consistency Regularization, IEEE RGS 2021, W. Wang et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/9380508)]

- Hyperspectral Image Super-Resolution Using Spectrum and Feature Context, IEEE TIM 2021, Qi Wang et al. [[Web](https://crabwq.github.io/)][[Pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9264655)]


### 3. Multispectral and Hyperspectral Image Fusion (MHF)
###### 1) Bayesian based approaches
- Blind Image Fusion for Hyperspectral Imaging with the Directional Total Variation, Inverse Problems, 2018, Leon Bungert et al.
[[PDF](https://arxiv.org/abs/1710.05705)]
[[Code](https://github.com/mehrhardt/blind_remote_sensing)]

- Bayesian sparse representation for hyperspectral image super resolution, CVPR2015, N. Akhtar et al.
[[PDF]( https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Akhtar_Bayesian_Sparse_Representation_2015_CVPR_paper.pdf)]
[[Code](http://openremotesensing.net/wp-content/uploads/2016/12/Supplementary.zip)]

- Hysure: A convex formulation for hyperspectral image superresolution via subspace-based regularization, TGRS2015, M. Simoes et al.
[[PDF](https://arxiv.org/abs/1411.4005)]
[[Code](https://github.com/alfaiate/HySure)]

- Hyperspectral and multispectral image fusion based on a sparse representation, TGRS2015, Q. Wei et al.
[[PDF](https://arxiv.org/pdf/1409.5729)]
[[Code](http://wei.perso.enseeiht.fr/demo/SparseFusion_2014-12-03.zip)]

- Bayesian fusion of multi-band images, Jstar2015, W. Qi et al.
[[PDF](http://wei.perso.enseeiht.fr/papers/WEI_JSTSP_final.pdf)]
[[Code](http://wei.perso.enseeiht.fr/demo/MCMCFusion.7z)]

- Noise-resistant wavelet-based Bayesian fusion of multispectral and hyperspectral images, TGRS2009, Y. Zhang et al.
[[PDF](https://ieeexplore.ieee.org/iel5/36/4358825/04967929.pdf)]

- Weighted Low-rank Tensor Recovery for Hyperspectral Image Restoration, arXiv2018, Yi Chang et al.
[[PDF](https://arxiv.org/pdf/1709.00192.pdf)]

- Enhanced Hyperspectral Image Super-Resolution via RGB Fusion and TV-TV Minimization, ICIP 2021, Marija Vella et al. [[PDF](https://arxiv.org/abs/2106.07066)][[Code](https://github.com/marijavella/hs-sr-tvtv)]

###### 2) Tensor based approaches
- Hyperspectral image superresolution via non-local sparse tensor factorization, CVPR2017, R. Dian et al.
[[PDF](https://sites.google.com/site/leyuanfang/cvpr-17)]

- Spatial–Spectral-Graph-Regularized Low-Rank Tensor Decomposition for Multispectral and Hyperspectral Image Fusion, Jstars2018, K. Zhang et al.
[[PDF](https://www.researchgate.net/publication/322559599_Spatial-Spectral-Graph-Regularized_Low-Rank_Tensor_Decomposition_for_Multispectral_and_Hyperspectral_Image_Fusion)]

- Fusing Hyperspectral and Multispectral Images via Coupled Sparse Tensor Factorization, TIP2108, S. Li et al.
[[PDF](https://drive.google.com/open?id=1ZJQB1RhjRO9JNTBVaNknk1DXjiWfg6Gd)]
[[Code](https://drive.google.com/open?id=12eleEjv7wKQxFCBUcIGkEl-wiUiJxwTv)]

- Hyperspectral Super-Resolution: A Coupled Tensor Factorization Approach, arXiv2018, Charilaos I. Kanatsoulis et al.
[[PDF](https://arxiv.org/pdf/1804.05307.pdf)]

- Nonlocal Patch Tensor Sparse Representation for Hyperspectral Image Super-Resolution, TIP2019, Yang Xu et al.
[[PDF](https://ieeexplore.ieee.org/abstract/document/8618436)]
[[Web](http://www.escience.cn/people/xuyangcs/paper.dhome)]

- Learning a Low Tensor-Train Rank Representation for Hyperspectral Image Super-Resolution, TNNLS2019, Renwei Dian et al.
[[PDF](https://ieeexplore.ieee.org/abstract/document/8603806)]
[[Web](https://github.com/renweidian/LTTR)]

- Nonnegative and Nonlocal Sparse Tensor Factorization-Based Hyperspectral Image Super-Resolution, IEEE TGRS2020, Wei Wan et al.
[[PDF](https://ieeexplore.ieee.org/abstract/document/9082892)]

- Nonlocal Coupled Tensor CP Decomposition for Hyperspectral and Multispectral Image Fusion, IEEE TGRS2020, Xu Yang et al.
[[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8835149)]

- Hyperspectral Super-Resolution via Coupled Tensor Ring Factorization, IEEE TGRS2020, Wei He et al.
[[PDF](https://arxiv.org/pdf/2001.01547.pdf)]

- Spatial-Spectral Structured Sparse Low-Rank Representation for Hyperspectral Image Super-Resolution, IEEE TIP2021, Jize Xue et al.,
[[PDF](https://ieeexplore.ieee.org/abstract/document/9356457)]

- Hyperspectral Images Super-Resolution via Learning High-Order Coupled Tensor Ring Representation, IEEE TNNLS 2020, Y. Xu et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/8948303)]

- Hyperspectral Image Superresolution Using Global Gradient Sparse and Nonlocal Low-Rank Tensor Decomposition With Hyper-Laplacian Prior, IEEE JStars 2021, Y. Peng et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/9417623)]

- Hyperspectral Image Superresolution via Structure-Tensor-Based Image Matting, IEEE JStars 2021, H. Gao et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/9508188)]


###### 3) Matrix factorization based approaches
- High-resolution hyperspectral imaging via matrix factorization, CVPR2011, R. Kawakami et al.
[[PDF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.3532&rep=rep1&type=pdf)]
[[Code](http://www.nae-lab.org/~rei/research/hh/index.html)]

- Coupled nonnegative matrix factorization unmixing for hyperspectral and multispectral data fusion, TGRS2012, N. Yokoya et al.
[[PDF](http://naotoyokoya.com/assets/pdf/NYokoyaTGRS2012.pdf)]
[[Code](http://naotoyokoya.com/assets/zip/CNMF_MATLAB.zip)]

- Sparse spatio-spectral representation for hyperspectral image super-resolution, ECCV2014, N. Akhtar et al.
[[PDF](http://openremotesensing.net/wp-content/uploads/2016/12/ECCV2014_Naveed.pdf)]
[[Code](http://openremotesensing.net/wp-content/uploads/2016/12/HSISuperRes.zip)]

- Hyper-sharpening: A first approach on SIM-GA data, Jstars2015, M. Selva et al.

- Hyperspectral super-resolution by coupled spectral unmixing, ICCV2015, C Lanaras.
 [[PDF](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lanaras_Hyperspectral_Super-Resolution_by_ICCV_2015_paper.pdf)]
[[Code](https://github.com/lanha/SupResPALM)]

- RGB-guided hyperspectral image upsampling, CVPR2015, H. Kwon et al.
[[PDF](https://pdfs.semanticscholar.org/2dfb/a20d04832e6ee7076c695f7bb99dcf1a3f02.pdf)]
[[Code](https://sites.google.com/site/rgbhsupsampling/)]

- Multiband image fusion based on spectral unmixing, TGRS2016, Q. Wei et al.
[[PDF](https://arxiv.org/abs/1603.08720)] 
[[Code](https://github.com/qw245/FUMI)]

- Hyperspectral image super-resolution via non-negative structured sparse representation, TIP2016, W. Dong, et al.
[[PDF](http://see.xidian.edu.cn/faculty/wsdong/Papers/Journal/NSSR_HSI_TIP16.pdf)] [[Code](http://see.xidian.edu.cn/faculty/wsdong/Code_release/NSSR_HSI_SR.rar)]

- Hyperspectral super-resolution of locally low rank images from complementary multisource data, TIP2016, M. A. Veganzones et al.
[[PDF](https://hal.archives-ouvertes.fr/hal-00960076/document)]

- Multispectral and hyperspectral image fusion based on group spectral embedding and low-rank factorization, TGRS2017, K. Zhang et al.

- Hyperspectral Image Super-Resolution Based on Spatial and Spectral Correlation Fusion, TRGS2018, C. Yi et al.

- Self-Similarity Constrained Sparse Representation for Hyperspectral Image Super-Resolution, TIP2108, X. Han et al.

- Exploiting Clustering Manifold Structure for Hyperspectral Imagery Super-Resolution, TIP2018, L. Zhang et al.
 [[Code](https://sites.google.com/site/leizhanghyperspectral/publications)]
 
- Hyperspectral Image Super-Resolution With a Mosaic RGB Image, TIP2018, Y. Fu et al.
[[PDF](https://ieeexplore.ieee.org/document/8410569/)]

- Fusing Hyperspectral and Multispectral Images via Coupled Sparse Tensor Factorization, TIP2018, S. Li et al.
[[PDF](https://drive.google.com/open?id=1ZJQB1RhjRO9JNTBVaNknk1DXjiWfg6Gd)][[Code](https://github.com/renweidian/CSTF)]

- Multispectral Image Super-Resolution via RGB Image Fusion and Radiometric Calibration, TIP2019, Zhi-Wei Pan et al.
[[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8543870)]
 [[Web](http://www.ivlab.org/panzhw.html)]
 
 - Hyperspectral Image Super-resolution via Subspace-Based Low Tensor Multi-Rank Regularization, TIP2019, Renwei Dian et al. [[PDF](https://drive.google.com/open?id=1k8o_uBwImUDkywMWEwFd-PFo6gB5Sv8U)]
 
 - Hyperspectral Image Super-Resolution With Optimized RGB Guidance, Ying Fu et al., CVPR2019. [[PDF](https://ieeexplore.ieee.org/document/8954237/)] 
 
 - Super-Resolution for Hyperspectral and Multispectral Image Fusion Accounting for Seasonal Spectral Variability, TIP2020, R.A. Borsoi et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8768351)]
 
 - A Truncated Matrix Decomposition for Hyperspectral Image Super-Resolution, TIP2020, Jianjun Liu et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9147021)]

- Adaptive Nonnegative Sparse Representation for Hyperspectral Image Super-Resolution, IEEE JStars 2021, X. Li et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/9399788)]
 
###### 4) Deep Learning based approaches
- Deep Residual Convolutional Neural Network for Hyperspectral Image Super-Resolution, ICIG2017, C. Wang et al.
[[PDF](http://www.ict.griffith.edu.au/~junzhou/papers/C_ICIG_2017.pdf)]


- SSF-CNN: Spatial and Spectral Fusion with CNN for Hyperspectral Image Super-Resolution, ICIP2018, X. Han et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8451142)]

- Deep Hyperspectral Image Sharpening, TNNLS2018, R. Dian et al.
[[PDF](https://drive.google.com/open?id=1FIyVL9c8jlDY3heEZ57nGvpSDZc0mkeT)]
[[Code](https://github.com/renweidian/DHSIS)]


- HSI-DeNet: Hyperspectral Image Restoration via Convolutional Neural Network, TGRS2018, Y. Chang et al.
[[Web](http://www.escience.cn/people/changyi/index.html)]

- Unsupervised Sparse Dirichlet-Net for Hyperspectral Image Super-Resolution, CVPR2018, Y. Qu et al.
[[PDF](https://arxiv.org/abs/1804.05042)]
[[Code](https://github.com/aicip/uSDN)]

- Deep Hyperspectral Prior: Denoising, Inpainting, Super-Resolution, arXiv2019, Oleksii Sidorov et al.
[[PDF](https://arxiv.org/abs/1902.00301)]
[[Code](https://github.com/acecreamu/deep-hs-prior)]
 
 - Multi-level and Multi-scale Spatial and Spectral Fusion CNN for Hyperspectral Image Super-resolution, ICCVW 2019, Xianhua Han et al.
[[PDF](http://openaccess.thecvf.com/content_ICCVW_2019/papers/PBDL/Han_Multi-Level_and_Multi-Scale_Spatial_and_Spectral_Fusion_CNN_for_Hyperspectral_ICCVW_2019_paper.pdf)]

 - Multispectral and Hyperspectral Image Fusion by MS/HS Fusion Net, CVPR2019, Xie Qi et al.
[[PDF](https://arxiv.org/pdf/1901.03281.pdf)]
 [[Web](https://scholar.google.com/citations?hl=zh-CN&user=2ZqIzTMAAAAJ&view_op=list_works&sortby=pubdate)]
 
  - Hyperspectral Image Reconstruction Using Deep External and Internal Learning,ICCV2019, Zhang Tao et al.
[[PDF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Hyperspectral_Image_Reconstruction_Using_Deep_External_and_Internal_Learning_ICCV_2019_paper.pdf)]
 [[Web](https://scholar.google.com/citations?hl=zh-CN&user=2ZqIzTMAAAAJ&view_op=list_works&sortby=pubdate)]
 
 - Deep Blind Hyperspectral Image Super-Resolution, IEEE TNNLS 2020, Lei Zhang et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/9136736)] 

  - Deep Recursive Network for Hyperspectral Image Super-Resolution, IEEE TCI2020, Wei Wei, et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9162463)][[Web](https://sites.google.com/site/leizhanghyperspectral/home)]

- Coupled Convolutional Neural Network With Adaptive Response Function Learning for Unsupervised Hyperspectral Super Resolution, IEEE TGRS 2020, K. Zheng et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/9141341)]

- Unsupervised Adaptation Learning for Hyperspectral Imagery Super-Resolution, CVPR 2020, L. Zhang et al. [[Pdf](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Unsupervised_Adaptation_Learning_for_Hyperspectral_Imagery_Super-Resolution_CVPR_2020_paper.html)]

- Cross-Attention in Coupled Unmixing Nets for Unsupervised Hyperspectral Super-Resolution, ECCV 2020, J. Yao et al. [[Pdf](https://link.springer.com/chapter/10.1007/978-3-030-58526-6_13)]
  
- Unsupervised Recurrent Hyperspectral Imagery Super-Resolution Using Pixel-Aware Refinement, IEEE TGRS2021, Wei Wei, et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9292466)][[Web](https://sites.google.com/site/leizhanghyperspectral/home)] 

- A Band Divide-and-Conquer Multispectral and Hyperspectral Image Fusion Method, IEEE TGRS 2021, Weiwei Sun et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/9363338)] 

- Hyperspectral Image Super-Resolution via Deep Progressive Zero-Centric Residual Learning, IEEE TIP 2021, Zhiyu Zhu et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/9298460)]

- Hyperspectral Image Super-Resolution via Deep Prior Regularization with Parameter Estimation, IEEE TCSVT 2021, X. Wang et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/9427207)]

- Hyperspectral Image Super-Resolution with Self-Supervised Spectral-Spatial Residual Network, RS 2021, W. Chen et al. [[Pdf](https://www.mdpi.com/2072-4292/13/7/1260)]

- Hyperspectral Image Super-Resolution via Deep Spatiospectral Attention Convolutional Neural Networks, IEEE TNNLS 2021, J. Hu et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/9449622)]

- Model-Guided Deep Hyperspectral Image Super-Resolution, IEEE TIP 2021, W. Dong et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/9429905)] [[Web](https://see.xidian.edu.cn/faculty/wsdong/Projects/MoG-DCN.htm)]

- Learning A 3D-CNN and Transformer Prior for hyperspectral Image Super-Resolution, arXiv 2021, Q. Ma et al. [[Pdf](https://arxiv.org/abs/2111.13923)] **This is our method and achieves state-of-the-art performance for Multispectral and Hyperspectral Image Fusion (MHF) task**



###### 5) Simulations registration and super-resolution approaches
 
- An Integrated Approach to Registration and Fusion of Hyperspectral and Multispectral Images, TRGS 2019, Yuan Zhou et al.

- Deep Blind Hyperspectral Image Fusion, ICCV2019, Wu Wang et al. [[PDF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Blind_Hyperspectral_Image_Fusion_ICCV_2019_paper.pdf)]

- Unsupervised and Unregistered Hyperspectral Image Super-Resolution With Mutual Dirichlet-Net, IEEE TGRS 2021, Y. Qu et al. [[Pdf](https://ieeexplore.ieee.org/abstract/document/9442804)]
 
================================================================================


================================================================================

#### Databases
- [CAVE dataset](http://www.cs.columbia.edu/CAVE/databases/multispectral/)
- [Harvard dataset](http://vision.seas.harvard.edu/hyperspec/explore.html)
- [iCVL dataset](http://icvl.cs.bgu.ac.il/hyperspectral/)
- [NUS datase](https://sites.google.com/site/hyperspectralcolorimaging/dataset/general-scenes)
- [NTIRE18 dataset](http://www.vision.ee.ethz.ch/ntire18/)
- [Chikusei dataset](https://www.sal.t.u-tokyo.ac.jp/hyperdata/)
- [Indian Pines, Salinas, KSC et al.](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

================================================================================

#### Image Quality Measurement
- Peak Signal to Noise Ratio (PSNR)
- Root Mean Square Error (RMSE)
- Structural SIMilarity index (SSIM)
- Spectral Angle Mapper (SAM)
- Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS)
- Universal Image Quality Index (UIQI)

