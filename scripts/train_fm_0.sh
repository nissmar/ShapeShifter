CFG='./configs/train_diffusion_0.yaml'

python src/diffusion/train_flow_matching.py -model_name "fighting-pillar" -level 0 -config $CFG
python src/diffusion/train_flow_matching.py -model_name "wood" -level 0 -config $CFG
python src/diffusion/train_flow_matching.py -model_name "small-town" -level 0 -config $CFG
python src/diffusion/train_flow_matching.py -model_name "ruined-tower" -level 0 -config $CFG
python src/diffusion/train_flow_matching.py -model_name "acropolis" -level 0 -config $CFG
python src/diffusion/train_flow_matching.py -model_name "house" -level 0 -config $CFG
python src/diffusion/train_flow_matching.py -model_name "stone-cliff" -level 0 -config $CFG
python src/diffusion/train_flow_matching.py -model_name "canyon" -level 0 -config $CFG
