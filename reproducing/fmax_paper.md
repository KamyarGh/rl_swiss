# Guide to reproducing the results from the fmax paper
Notes:
- First appropriately modify rlkit/launchers/config.py
- run_experiment.py calls srun which is a SLURM command. You can use the `--nosrun` flag to not use SLURM and use your local machine instead.
- The expert demonstrations and state marginal data used for imitation learning experiments can be found at [THIS LINK](https://drive.google.com/drive/folders/1jwKb5FjFtAlvBUDdHiHJN0i7PsBCthfg?usp=sharing). To use them please download them and modify the paths in expert_demos_listing.yaml.
- The yaml files describe the experiments to run and have three sections:
..* meta_data: general experiment and resource settings
..* variables: used to describe the hyperparameters to search over
..* constants: hyperparameters that will not be searched over
- The conda env specs are in rl_swiss_conda_env.yaml. You can refer to [THIS LINK](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-from-file) for notes on how to set up your conda environment using the rl_swiss_conda_env.yaml file.
- You need to have Mujoco and mujoco-py installed.
- Due to a minor dependency on rllab, you would have to also install rllab. I will try to remove this dependency in future versions. The dependency is that run_experiment.py calls build_nested_variant_generator which uses something from rllab.

## Reproducing Imitation Learning Results
### Training Expert Policies
If you would like train your own expert policies with Soft-Actor-Critic you can for example run:
```bash
python run_experiment.py --nosrun -e exp_specs/sac.yaml
```
To train a policy for a different environment, add your environment to the file in rlkit/envs/envs_dict and replace the name of your environment in the env_specs->env_name field in sac.yaml.

### Generating Demonstrations Using an Expert Policy
Modify exp_specs/gen_exp_demos.yaml appropriately and run:
```bash
python run_experiment.py --nosrun -e exp_specs/gen_exp_demos.yaml
```

### Training F/AIRL
Put the path to your expert demos in expert_demos_listing.yaml. Modify exp_specs/adv_irl.yaml appropriately and run:
```bash
python run_experiment.py --nosrun -e exp_specs/adv_irl.yaml
```
For all four imitation learning domains we used the same hyperparameters except grad_pen_weight and reward_scale which were chosen with a small hyperparameter search.

### Training BC
```bash
python run_experiment.py --nosrun -e exp_specs/bc.yaml
```

### Training DAgger
```bash
python run_experiment.py --nosrun -e exp_specs/dagger.yaml
```

## Reproducing State-Marginal-Matching Results
### Generating the target state marginal distributions
This is a little messy and I haven't gotten to cleaning it up yet. All the scripts that you see in the appendix of the paper can be found in these three files: data_gen/point_mass_data_gen.py, data_gen/fetch_state_marginal_matching_data_gen.py, data_gen/pusher_data_gen.py.

### Training SMM Models
When you run any of these scripts, at every epoch in the log directory of the particular experiment images will be save to demonstrate the state distribution of the policy obtained at that point. For more information about the visualizations you can have a look at the log_visuals function implemented for each of the environments used in the state-marginal-matching experiments.
```bash
python run_experiment.py --nosrun -e exp_specs/point_mass_spiral_adv_smm.yaml
```
```bash
python run_experiment.py --nosrun -e exp_specs/point_mass_infty_adv_smm.yaml
```
```bash
python run_experiment.py --nosrun -e exp_specs/pusher_trace_adv_smm.yaml
```
```bash
python run_experiment.py --nosrun -e exp_specs/pusher_push_adv_smm.yaml
```
```bash
python run_experiment.py --nosrun -e exp_specs/fetch_push_adv_smm.yaml
```
