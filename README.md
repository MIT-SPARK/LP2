![](https://github.com/MIT-SPARK/LP2/blob/main/assets/project_page_title.gif?raw=true "Our method, LP^2, predicts a spatio-temporal distribution over long-term (up to 60 s) human trajectories in complex environments by reasoning about their interactions with the scene, represented as a 3D Dynamic Scene Graph.")

# LP<sup>2</sup>: Language-based Probabilistic Long-term Prediction
This repository contains the code for LP<sup>2</sup>, our approach for Long-term Human Trajectory Prediction using 3D Dynamic Scene Graphs.

> This project was supported by Amazon, Lockheed Martin, and the Swiss National Science Foundation (SNSF).

# Table of Contents
**Credits**
* [Paper](#paper)
* [Video](#video)

**Setup**
* [OpenAI API Setup](#openai-api-setup)
* [Installation](#installation)
* [Dataset](#dataset)


 **Examples**
* [Running LP<sup>2</sup>](#running)
* [Visualization](#visualization) 

# Paper
If you find this useful for your research, please consider citing our paper:

* Nicolas Gorlo, Lukas Schmid, and Luca Carlone, "**Long-Term Human Trajectory Prediction using 3D Dynamic Scene Graphs**", in _IEEE Robotics and Automation Letters_, vol. 9, no. 11, pp. 10978-10985, 2024. [ [Paper](https://ieeexplore.ieee.org/document/10720207) | [Preprint](https://arxiv.org/abs/2405.00552) | [Video](https://www.youtube.com/watch?v=mzumT3T0dYw) ]
  ```bibtex
   @ARTICLE{Gorlo2024LP2,
    author={Gorlo, Nicolas and Schmid, Lukas and Carlone, Luca},
    journal={IEEE Robotics and Automation Letters},
    title={Long-Term Human Trajectory Prediction Using 3D Dynamic Scene Graphs},
    year={2024},
    volume={9},
    number={12},
    pages={10978-10985},
    doi={10.1109/LRA.2024.3482169}
    }
  ```
# Video

An overview of our approach is available on [YouTube](https://www.youtube.com/watch?v=mzumT3T0dYw):

[<img src=https://github.com/MIT-SPARK/LP2/assets/36043993/0dd28295-3b72-468b-8420-56477f910e8b alt="Youtube Video">](https://www.youtube.com/watch?v=mzumT3T0dYw)

# OpenAI API Setup 
Our method uses the OpenAI API to predict interactions between humans and the scene. To use the OpenAI API, you need to create an account and obtain an API key. You can find more information on the OpenAI API [here](https://platform.openai.com/docs/overview).
Specifically, you need to setup an account and buy Pay-as-you-go credits to be able to use the API.
Our code relies on the following environment variables to access the OpenAI API. These will be associated with your personal account:
```bash
export OPENAI_API_KEY=???
export OPENAI_API_ORG=???
```
Once your account is set up, the keys are available in the [OpenAI profile tab](https://platform.openai.com/settings/profile?tab=api-keys).

# Installation 
1. Clone the repository
```bash
git clone git@github.com:MIT-SPARK/LP2.git
cd LP2
```
2. Create a python 3.10 environment (using [pyenv](https://github.com/pyenv/pyenv))
```bash
pyenv install 3.10.15
pyenv virtualenv 3.10.15 LP2
pyenv activate LP2
```
3. Install the required packages
```bash
pip install -r requirements.txt
```

# Dataset
To download the dataset, run:
```bash
python scripts/download_data.py
```

# Running
The main script of our codebase will first run the pipeline for each trajectory in the selected split of the dataset. Then, it will evaluate the predictions and save the results in the `output` folder. To add additional steps or load checkpoints created while running the method, specify the corresponding parameters in the config file (e.g., `'global_config/global/load_checkpoints'`, `'project_config/global/animate'`).
Feel free to adjust the configurations in the `config` folder or choose another scene config to select the part of the dataset to predict on. 
The following command will run the pipeline for the LP<sup>2</sup> method on the office scene.

```bash
python src/lhmp/main.py --run_pipeline "y" \
                            --global_config_file "config/global_config.yaml" \
                            --method_config_file "config/method_configs/project_config_LP2.yaml" \
                            --scene_config_file "config/scene_configs/scene_config_office.yaml"
```

To run LP<sup>2</sup> on your own data, you can bring the data into the same format as the downloaded data and create a new scene config file specifying the paths to the data.
Note, that only the parameters `scene_graph_path`, `room_labels_path`, and `trajectory_dir` are required to run the method. The hierarchical 3D scene graph is in the [Spark-DSG](https://github.com/MIT-SPARK/Spark-DSG) json format.

# Visualization
To plot the negative log likelihood of predictions, use the following command:
```bash
python3 scripts/visualize_all.py --methods "LP2" "LP2_instance" \
                                 --scenes "office" "home" \
                                 --n_past_interactions 2 \
                                 --time_lim 60.0 \
                                 --plot_std False
```
