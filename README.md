# EGIC (TensorFlow 2)

> **EGIC: Enhanced Low-Bit-Rate Generative Image Compression Guided by Semantic Segmentation** <br>

## Abstract

We introduce EGIC, an enhanced generative image compression method that allows traversing the distortion-perception 
curve efficiently from a single model. EGIC is based on two novel building blocks: i) OASIS-C, a conditional pre-trained semantic 
segmentation-guided discriminator, which provides both spatially and semantically-aware gradient feedback to the generator, 
conditioned on the latent image distribution, and ii) Output Residual Prediction (ORP), a retrofit solution for multi-realism 
image compression that allows control over the synthesis process by adjusting the impact of the residual between an MSE-optimized 
and GAN-optimized decoder output on the GAN-based reconstruction. Together, EGIC forms a powerful codec, outperforming state-of-the-art 
diffusion and GAN-based methods (e.g., HiFiC, MS-ILLM, and DIRAC-100), while performing almost on par with VTM-20.0 on the distortion end. 
EGIC is simple to implement, very lightweight, and provides excellent interpolation characteristics, which makes it a promising candidate 
for practical applications targeting the low bit range.

<div align=center>
<img src="./doc/assets/teaser_clic2020_v2.png" width="70%">
</div>


<p align="center"><em>Distortion-perception comparison. Top left is best.</em></p>