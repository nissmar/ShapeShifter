

# ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion [CVPR 2025]

[Nissim Maruani](https://nissmar.github.io)<sup>1,2</sup>, [Wang Yifan](https://yifita.netlify.app)<sup>3</sup>, [Matthew Fisher](https://techmatt.github.io)<sup>3</sup>, [Pierre Alliez](https://team.inria.fr/titane/pierre-alliez/)<sup>1,2</sup>, [Mathieu Desbrun](https://pages.saclay.inria.fr/mathieu.desbrun/)<sup>1,4</sup>.

<sup>1</sup>Inria, <sup>2</sup>Université Côte d'Azur, <sup>3</sup>Adobe Research, <sup>4</sup>École polytechnique


<div align="center">
  <img src="media/shapeshifter-summary.svg" alt="ShapeShifter Summary" width="80%">
</div>

Please see our [project page](https://nissmar.github.io/projects/shapeshifter/) for video presentation & 3D meshes.

## News

- `2025-02` Accepted at CVPR 2025 🚀🚀🚀

## Getting Started

1. Start by cloning the repository and *fVDB* submodule:

```shell
git clone --recursive https://github.com/nissmar/ShapeShifter.git
```

2. Create the `shapeshifter` conda environment:
````shell
conda env create -f dev_env.yml
conda activate shapeshifter
````


3. Our code requires building *fVDB*, which can take a while (please refer to the original [README](openvdb/fvdb/README.md) for more details). Run: 
```shell
cd openvdb/fvdb
export MAX_JOBS=$(free -g | awk "/^Mem:/{jobs=int($4/2.5); if(jobs<1) jobs=1; print jobs}")
pip install .
cd ../..
```
 
## Data and Pretrained Models

Download the data and/or pre-trained checkpoints [here](https://drive.google.com/drive/u/2/folders/1na3wXBaAqz21lxmbItQHOXn4tSJOX6TQ) and place them at the source of this repository.


## Quickstart

Run the notebook [src/diffusion/sample_diffusion.ipynb](src/diffusion/sample_diffusion.ipynb)

## Evaluation

Sample diffusion models and extract pointclouds
```shell
python src/diffusion/sample_diffusion.py
``` 

Mesh the pointclouds 
```shell
python src/meshing/mesh_with_poisson_or_apss.py
```
 
Compute the metrics with [src/eval/eval_metrics.ipynb](src/eval/eval_metrics.ipynb)

## Train 

### Parallel training 

All of the levels (upsampler + diffusion) are independent of each other and can be trained in parallel. Training the upsampler + the diffusion model roughly takes 12 minutes per level. 

For the first level:
```shell
python src/diffusion/train_diffusion.py -model_name "acropolis" -level 0 -config "./configs/train_diffusion_0.yaml"
```
For the subsequent levels:
```shell
python src/diffusion/train_upsamplers.py -model_name "acropolis" -level 1 -config "./configs/train_upsampler.yaml" ; 
python src/diffusion/train_diffusion.py -model_name "acropolis" -level 1 -config "./configs/train_diffusion_up.yaml"
```

### Sequential training

Easier but longer:
```shell
bash scripts/train_upsamplers.sh ;
bash scripts/train_diffusion_0.sh ;
bash scripts/train_diffusion_up.sh
```