

# ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion [CVPR 2025]

[Nissim Maruani](https://nissmar.github.io)<sup>1,2</sup>, [Wang Yifan](https://yifita.netlify.app)<sup>3</sup>, [Matthew Fisher](https://techmatt.github.io)<sup>3</sup>, [Pierre Alliez](https://team.inria.fr/titane/pierre-alliez/)<sup>1,2</sup>, [Mathieu Desbrun](https://pages.saclay.inria.fr/mathieu.desbrun/)<sup>1,4</sup>.

<sup>1</sup>Inria, <sup>2</sup>Université Côte d'Azur, <sup>3</sup>Adobe Research, <sup>4</sup>École polytechnique

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
export MAX_JOBS=$(free -g | awk '/^Mem:/{jobs=int($4/2.5); if(jobs<1) jobs=1; print jobs}')
pip install .
cd ../..
```
 
## Process Your Own Data

TODO

## Data and Pretrained Models
 
TODO

## Evaluation



### Sample

Sample and extract pointclouds
```shell
python src/diffusion/sample_diffusion.py
``` 

### Mesh

Mesh the pointclouds 
```shell
python src/meshing/mesh_with_poisson_or_apss.py
```

### Evaluation
 
Run notebooks `src/eval/eval_metrics.ipynb`

## Train 

### Train Upsamplers

`bash scripts/train_upsamplers.sh`

### Train Diffusion

`bash scripts/train_diffusion_0.sh`

`bash scripts/train_diffusion_up.sh`

bash scripts/train_upsamplers.sh ; bash scripts/train_diffusion_0.sh ; bash scripts/train_diffusion_up.sh