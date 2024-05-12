# SAP-samples/repository-template
This default template for SAP Samples repositories includes files for README, LICENSE, and .reuse/dep5. All repositories on github.com/SAP-samples will be created based on this template.

# Containing Files

1. The LICENSE file:
In most cases, the license for SAP sample projects is `Apache 2.0`.

2. The .reuse/dep5 file: 
The [Reuse Tool](https://reuse.software/) must be used for your samples project. You can find the .reuse/dep5 in the project initial. Please replace the parts inside the single angle quotation marks < > by the specific information for your repository.

3. The README.md file (this file):
Please edit this file as it is the primary description file for your project. You can find some placeholder titles for sections below.

# Graph Learning for Code Vulnerability Detection
<!-- Please include descriptive title -->

<!--- Register repository https://api.reuse.software/register, then add REUSE badge:
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/REPO-NAME)](https://api.reuse.software/info/github.com/SAP-samples/REPO-NAME)
-->

## Description
This repository contains sample code to reproduce the research done for the bachelor thesis _"Deep Learning-Based Code Vulnerability Detection: A New Perspective"_ at SAP Security Research. 

The repository implements an GNN evaluation pipeline including cross-validation as well as pretraining schedules.

## Download and Installation

To run the experiments, the [DiversVul dataset](https://github.com/wagner-group/diversevul) (Chen, Yizheng, et al. 2023) must be downloaded, graphs need to be parsed with the [cpg](https://github.com/Fraunhofer-AISEC/cpg) tool and python packages in ``0_install`` are required. Further, scripts in ``codegraphs/diversevul/`` produce intermediate pickle files for cross-validation and filtering large and small graphs, which ``CodeGraphDataset.py`` requires to load the datasets.

## Running the experiments

All configuration files can be found in ``configs/``. By switching out the filename in ``1_train.py`` different models can be run. ``2_helper_get_best_run.py`` summarizes results from cross-validation.

The main test results are produced with the ``configs/7_*`` and ``configs/9_*`` files.

Visualizations from the paperare made with scripts in ``utils/``

Different models as well as the training script are specified in ``models``.

## Known Issues
No known issues.

## How to obtain support
[Create an issue](https://github.com/SAP-samples/<repository-name>/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.
