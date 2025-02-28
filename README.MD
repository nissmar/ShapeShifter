

# ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion [CVPR 2025]

[Nissim Maruani](https://nissmar.github.io)<sup>1</sup>, [Wang Yifan](https://yifita.netlify.app)<sup>2</sup>, [Matthew Fisher](https://techmatt.github.io)<sup>2</sup>, [Pierre Alliez](https://team.inria.fr/titane/pierre-alliez/)<sup>1</sup>, [Mathieu Desbrun](https://pages.saclay.inria.fr/mathieu.desbrun/)<sup>1,3</sup>.

<sup>1</sup>Inria, <sup>2</sup>Adobe Research, <sup>3</sup>École polytechnique

## News

- `2025-02` Accepted at CVPR 2025 🚀🚀🚀

## Getting Started



### Train Upsamplers

`bash scripts/train_upsamplers.sh `

### Train Diffusion

`bash scripts/train_diffusion_0.sh`

`bash scripts/train_diffusion_up.sh`


### Sample

Sample and extract pointclouds
`python src/diffusion/eval.py` 

### Mesh

Mesh the pointclouds 

`python src/meshing/mesh_with_poisson_or_apss.py`

### Evaluation
 
Run notebooks

`src/eval/eval_metrics.ipynb`
