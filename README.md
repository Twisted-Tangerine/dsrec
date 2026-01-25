# DSRec: Dual-Space Sequential Recommendation

The implementation of the submission "DSRec: Mitigating Double Noise in LLM-Enhanced Sequential Recommendation via Dual-Space Embedding Adaptation".

## Configure the environment

Versions of our hardware and software equipments:
- Hardware:
  - GPU: RTX 4090
  - Cuda: 11.8
- Software:
  - Python: 3.9
  - Pytorch: 2.1.2

And you can conduct pip install the `requirements.txt` to configure the environment.
```sh
pip install -r requirements.txt
```
By the way, we recommend you install the `tmux`

## Dataset

## Generation

### semantic embedding

### collaborative embedding
```sh
bash scripts/general_trip.sh > ./results/id_trip.log 2>&1
bash scripts/general_yelp.sh > ./results/id_yelp.log 2>&1
bash scripts/general_grocery.sh > ./results/id_grocery.log 2>&1
```

## Adaptation
```sh
bash scripts/train_trip.sh > ./results/trip.log 2>&1
bash scripts/train_yelp.sh > ./results/yelp.log 2>&1
bash scripts/train_grocery.sh > ./results/grocery.log 2>&1
```

The checkpoint will be saved in the folder `saved/`.