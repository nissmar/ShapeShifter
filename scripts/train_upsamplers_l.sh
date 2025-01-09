CFG='./configs/train_upsampler.yaml'

python src/diffusion/train_upsamplers_colors.py -model_name "fighting-pillar" -level 5 -config $CFG
python src/diffusion/train_upsamplers_colors.py -model_name "wood" -level 5 -config $CFG
python src/diffusion/train_upsamplers_colors.py -model_name "small-town" -level 5 -config $CFG
python src/diffusion/train_upsamplers_colors.py -model_name "ruined-tower" -level 5 -config $CFG
python src/diffusion/train_upsamplers_colors.py -model_name "acropolis" -level 5 -config $CFG
python src/diffusion/train_upsamplers_colors.py -model_name "house" -level 5 -config $CFG
python src/diffusion/train_upsamplers_colors.py -model_name "stone-cliff" -level 5 -config $CFG
python src/diffusion/train_upsamplers_colors.py -model_name "canyon" -level 5 -config $CFG