# EGIC (TensorFlow 2)

> **EGIC: Enhanced Low-Bit-Rate Generative Image Compression Guided by Semantic Segmentation** <br>
> Nikolai Körber, Eduard Kromer, Andreas Siebert, Sascha Hauke, Daniel Mueller-Gritschneder <br>
> ArXiv 2309.03244

## Abstract

We introduce EGIC, a novel generative image compression method that allows traversing the distortion-perception
curve efficiently from a single model. Specifically, we propose an implicitly encoded variant of image interpolation
that predicts the residual between a MSE-optimized and
GAN-optimized decoder output. On the receiver side, the
user can then control the impact of the residual on the
GAN-based reconstruction. Together with improved GANbased building blocks, EGIC outperforms a wide-variety of
perception-oriented and distortion-oriented baselines, including HiFiC, MRIC and DIRAC, while performing almost
on par with VTM-20.0 on the distortion end. EGIC is simple
to implement, very lightweight (e.g. 0.18× model parameters compared to HiFiC) and provides excellent interpolation characteristics, which makes it a promising candidate
for practical applications targeting the low bit range.

<div align=center>
<img src="./doc/assets/teaser_clic2020.png" width="70%">
</div>


<p align="center"><em>Distortion-perception comparison. Top left is better.</em></p>