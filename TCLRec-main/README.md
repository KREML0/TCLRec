# TCLRec: A Time-Aware Enhanced Contrastive Learning Framework for Mitigating Popularity Bias in Sequential Recommender Systems
This is the implementation of the submission "A Time-Aware Enhanced Contrastive Learning Framework for Mitigating Popularity Bias in Sequential Recommender Systems".
## Configuration of the environment
The hardware and software we used are listed below to facilitate the environment's configuration. The detailed environment setting can be found in the `requirements.txt`. You can use pip install to reproduce the environment.
- Hardware:
  - GPU: one NVIDIA A10
  - CUDA: 11.8
  
- Software:
  - Python: 3.10.13
  - Pytorch: 2.1.1 + cu118
  
- Usage
  - Install Causal Conv1d
    - `pip install causal-conv1d==1.1.3.post1`
    
  - Install Recbole
    - `pip install recbole==1.2.0`
    
  - Install Mamba
    - `pip install mamba-ssm==1.1.4`
  
  - Install DGL
    
    - `pip install dgl-cu116==0.9.1`
##  Datasets
The procedures for preprocessing the datasets are listed as follows:
- The raw datasets should be preprocessed using the Conversion tool provided by `https://github.com/RUCAIBox/RecSysDatasets`. After you acquire the atomic files, please put them into `dataset/<XXX/XXX/XXX/ml-1m>`. 
- Or you can directly download the atomic files of these datasets using the Baidu disk link provided by Recbole: `https://github.com/RUCAIBox/RecSysDatasets`.

## Model Training
- You can directly run the `model/run.py` to reproduce the training procedure.
