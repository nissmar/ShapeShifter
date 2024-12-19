


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
