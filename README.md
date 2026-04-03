# RADIO-ViPE: Online Tightly Coupled Multi-Modal Fusion for Open-Vocabulary Semantic SLAM in Dynamic Environments

**Contributors**:  Biomechatronics and Energy-Efficient Robotics (BE2R) Lab, ITMO University.

![alt text](assets/pipeline.jpg)

**Full Abstract**: We present **RADIO-ViPE** (**R**educe **A**ll **D**omains **I**nto **O**ne --- **Vi**deo **P**ose **E**ngine), an online semantic SLAM system that enables geometry-aware open-vocabulary grounding, associating arbitrary natural language queries with localized 3D regions and objects in dynamic environments. Unlike existing approaches that require calibrated, posed RGB-D input, RADIO-ViPE operates directly on raw monocular RGB video streams, requiring no prior camera intrinsics, depth sensors, or pose initialization. The system tightly couples multi-modal embeddings — spanning vision and language — derived from agglomerative foundation models (e.g., RADIO) with geometric scene information. This vision-language-geometric fusion is optimized within adaptive robust kernels, designed to handle both actively moving objects and agent-displaced scene elements (e.g., furniture rearranged during ego-centric session). Experiments demonstrate that RADIO-ViPE achieves state-of-the-art results on the dynamic TUM-RGBD benchmark while maintaining competitive performance against offline open-vocabulary methods that rely on calibrated data and static scene assumptions. RADIO-ViPE bridges a critical gap in real-world deployment, enabling robust open-vocabulary semantic grounding for autonomous robotics, AR/VR applications, and unconstrained in-the-wild video streams.

![Demo](assets/demo.gif)

## Installation
### Docker
```bash
# Build new docker image
make build

# Run docker image
make DATA_DIR={YOUR_DATA_DIR} run

# Inside docker
pip install --no-build-isolation -e .


## Usage

Example usages:

```bash
# Running the full pipeline.
python run.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO_OR_DIR_PATH

# Running the pose-only pipeline without depth estimation.
python run.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO_OR_DIR_PATH pipeline.post.depth_align_model=null
```

## Evaluation
- Soon will be documented and updated.

## Acknowledgments

ViPE is built on top of many great open-source research projects and codebases. Some of these include (not exhaustive):
- [RAD-SEG](https://arxiv.org/abs/2511.19704)
- [KM-ViPE](https://arxiv.org/abs/2512.01889)
- [RayFronts](https://arxiv.org/abs/2504.06994)
- [ViPE](https://github.com/nv-tlabs/vipe?tab=readme-ov-file)
- [RADIO](https://arxiv.org/abs/2601.17237)
- [DINOv3](https://github.com/facebookresearch/dinov3)
- [Talk2DINO](https://github.com/lorebianchi98/Talk2DINO)
- [RVWO](https://github.com/be2rlab/rvwo)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)



## License

This project will download and install additional third-party **models and softwares**. Note that these models or softwares are not distributed by NVIDIA. Review the license terms of these models and projects before use. This source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).
