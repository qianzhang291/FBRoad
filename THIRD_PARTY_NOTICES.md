# Third-Party Notices

Original repository code and scripts are provided under the [MIT License](LICENSE).
Bundled or adapted third-party components retain their original copyright notices
and upstream terms as summarized below.

| Component | Upstream source | Location and license |
|---|---|---|
| CMMPNet network, basic blocks, loader, augmentation, Dice/BCE | `liulingbo918/CMMPNet` (`8da4d0bab659182b11da62116635ff6bf13684ae`) | `src/fbroad/_vendor/cmmpnet/` and the adapted region forward in `src/fbroad/core_model.py`; upstream repository does not specify a license file |
| Four-direction decoder | `suniique/Leveraging-Crowdsourced-GPS-Data-for-Road-Extraction-from-Aerial-Imagery` (`b174045888b9b7daf2c61b03fa6f922048480863`) | Included in `src/fbroad/_vendor/cmmpnet/networks/basic_blocks.py`; MIT, copyright 2019 suniique; see `licenses/FourDirection.txt` from that frozen revision |
| CMX Feature Fusion Module (FFM) | `huaaaliu/RGBX_Semantic_Segmentation` (`e251d860aebc2f583a6c4919877e6bebe7f1aff3`) | `src/fbroad/cmx_ffm.py`; MIT (`licenses/CMX.txt`) |
| SegFix HRNet2x-W20 | `openseg-group/openseg.pytorch` (`aefc75517b09068d7131a69420bc5f66cb41f0ee`) | `src/fbroad/_vendor/segfix/`; MIT (`licenses/SegFix.txt`) |
| EMA (`ModelEmaV3`) | `huggingface/pytorch-image-models` (`timm`) | `src/fbroad/reference_ema.py`; Apache-2.0 (`licenses/timm.txt`) |
| Pixel contrast and projector | Adapted from `BubbleSai/DICN` (`a949a22f0ee381c90075c14092c97dcc933bda91`) | `src/fbroad/contrast.py`; research reimplementation with device/label guards |
| Binary Lovász hinge | `bermanmaxim/LovaszSoftmax` (`7d48792d35a04d3167de488dd00daabbccd8334b`) | `src/fbroad/lovasz_losses.py`; MIT (`licenses/LovaszSoftmax.txt`) |

PyTorch, torchvision, NumPy, OpenCV, SciPy, scikit-image, yacs and PyYAML are
external runtime dependencies under their respective licenses and are not bundled.
BJRoad imagery, ground-truth masks, rendered GPS heatmaps and model weights are not
redistributed in this repository.
