# DeepMove
PyTorch implementation of WWW'18  paper-DeepMove: Predicting Human Mobility with Attentional Recurrent Networks [link](https://dl.acm.org/citation.cfm?id=3178876.3186058)

# Datasets
All datasets are open-source:
- Foursquare-NYC(2. NYC and Tokyo Check-in Dataset) https://sites.google.com/site/yangdingqi/home/foursquare-dataset
- Porto Taxi https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data


# Requirements
- Python 3.8.0
- Pytorch 1.7.0
- Numpy 1.24.3

# Project Structure
- /codes
    - main.py
    - model.py # define models
    - sparse_traces.py # foursquare data preprocessing 
    - train.py # define tools for train the model
- /pretrain
    - /simple
        - res.m # pretrained model file
        - res.json # detailed evaluation results
        - res.txt # evaluation results
    - /simple_long
    - /attn_local_long
    - /attn_avg_long_user
- /data # preprocessed Foursquare and Proto data (pickle file)
- /docs # paper and presentation file
- /resutls # the default save path when training the model

# Usage
1. Load a pretrained model:
> ```python
> python main.py --model_mode=attn_avg_long_user --pretrain=1
> ```

The codes contain four network model (simple, simple_long, attn_avg_long_user, attn_local_long) and a baseline model (Markov). The parameter settings for these model can refer to their res.txt file.

Foursquare-NYC
|model_in_code | model_in_paper | top-1 accuracy (pre-trained)|
:---: |:---:|:---:
|markov | markov | 0.082|
|simple | RNN-short | 0.096|
|simple_long | RNN-long | 0.118|
|attn_avg_long_user | Ours attn-1 | 0.133|
|attn_local_long | Ours attn-2 | 0.145|

Foursquare-NYC
|model_in_code | model_in_paper | top-1 accuracy (pre-trained)|
:---: |:---:|:---:
|markov | markov | 0.00037|
|simple | LSTM-short | 0.0011|
|simple_long | LSTM-long | 0.0013|
|attn_avg_long_user | Ours attn-1 | TBD|
|attn_local_long | Ours attn-2 | TBD|

2. Train a new model:
> ```python
> python main.py --model_mode=attn_avg_long_user --pretrain=0
> ```

Other parameters (refer to main.py):
- for training: 
    - learning_rate, lr_step, lr_decay, L2, clip, epoch_max, dropout_p
- model definition: 
    - loc_emb_size, uid_emb_size, tim_emb_size, hidden_size, rnn_type, attn_type
    - history_mode: avg, avg, whole

# Others
You can regard simple and simple_long model setting as the basic LSTM/GRU/RNN.

Difference:
- Simple: Historical data is generated separately for each session, and specific processing can be performed on the historical data as needed.
- Simple_long: A long continuous sequence is generated without distinguishing between different sessions.