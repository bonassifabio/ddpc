## About
This repository implements the Data-Driven Predictive Control (DDPC) algorithms described in the paper
> Per Mattsson, Fabio Bonassi, Valentina Breschi, Thomas B. Schön, “_On the equivalence of direct and indirect data-driven predictive control approaches_,” 2024, arXiv preprint: 2403.05860 [[link]](https://arxiv.org/abs/2403.05860)

If you use this code, or otherwise found our work valuable, please cite the following paper

```
@article{mattsson2024equivalence,
      title={On the equivalence of direct and indirect data-driven predictive control approaches}, 
      author={Per Mattsson and Fabio Bonassi and Valentina Breschi and Thomas B. Schön},  
      journal={arXiv preprint arXiv:2403.05860},
      year={2024},
}
```

### Requirements
This code was developed on a Mac running Python 3.11, NumPy 12.6, cvxpy 1.4. 
Other requirements are listed in [requirements.txt](./requirements.txt).

Installation:
```
pip install -r requirements.txt
```

### Repository structure
```
gddpc/                      Source code for the implemented methods
  controller.py             Implementation of DDPC controllers
  system.py                 Implementation of the benchmark system
  utils.py                  Utility functions
analysis_equivalence.ipynb  A Jupyter Notebook analyzing the equivalence between the direct DeePC formulation and the indirect one
analysis_openloop.ipynb     A Jupyter Notebook analyzing the implemented DDPC methods
analysis_training.ipynb     A Jupyter Notebook analyzing the results of the test campaign (performances vs training size)
default_campaign.yaml       Default hyperparameters of the DDPCs
openloop_campaign.py        Python file running an intensive test campaign
openloop_test.py            Python file running open-loop test
```
