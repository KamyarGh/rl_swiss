*All of the things mentioned in the implemented section are not yet implemented in the refactored version. Hopefully will be done by the end of the weekend*

**Important Notes**

This repository (rlswiss) has been extended from the August 2018 version of [rlkit](https://github.com/vitchyr/rlkit). Since then the design approaches of rlswiss and rlkit have deviated quite a bit, and it is for this reason that we are releasing rlswiss as a separate repository. *If you find this repository useful for your research/projects, please cite this repository as well as [rlkit](https://github.com/vitchyr/rlkit).*

# rlswiss
**Reinforcement Learning (RL) and Learning from Demonstrations (LfD) framework for the single task as well as meta-learning settings.**

Our goal throughout has been to make it very efficient to implement new ideas quickly and cleanly. The core infrastructure is learning-framework-agnostic (PyTorch, Tf, etc.), however current implementations of specific algorithms are all in PyTorch.

Implemented RL algorithms:
- Soft-Actor-Critic (SAC)

Implemented LfD algorithms:
- Adversarial methods for Inverse Reinforcement Learning
    - AIRL / GAIL / FAIRL / Discriminator-Actor-Critic
- Behaviour Cloning
- DAgger

Implemented Meta-RL algorithms:
- RL with observed task parameters

Implemented Meta-LfD algorithms:
- SMILe
- Meta Behaviour Cloning
- Meta DAgger
