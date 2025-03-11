python src/diffusion/train_diffusion.py -model_name "$1" -level 0 -config './configs/train_diffusion_0.yaml'
for i in $(seq 1 4);
do
    python src/diffusion/train_upsamplers.py -model_name "$1" -level $i -config './configs/train_upsampler.yaml'
    python src/diffusion/train_diffusion.py  -model_name "$1" -level $i -config './configs/train_diffusion_up.yaml'
done

