# Readme for unsupervised image registration inference


This repository is created for Unsupervised multimodal image registration with deep learning for biomedical microscopy 

<details>
<summary>Install instructions </summary>
## Setup a python enviroment
create a virutal enviroment with python version 3.6

activate the virtualenv
pip install  -r requirements.txt

## download models
Download models for cut and SuperPoint

## download exmaple images and preprocess them

</details>

<details>
<summary>Run</summary>


## Run

Start the pipeline with arguments:
Cut model path: where latest_net.G is located
SuperPoint model path, where saved_model.pb is located
Image A path
Image B path



### Example

'''
python3 run_pipeline.py "./models/cut/cut_unaligned_resize/" "./models/sp/sp_v6/" "./Images/A/p1_wA1_t1_m9_c1_z0_l1_o0_1.png" "Images/B/p1_wA1_t1_m9_c1_z0_l1_o0_1.png"
'''

</details>