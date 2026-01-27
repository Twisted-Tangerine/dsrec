# DSRec: Dual-Space Sequential Recommendation

The implementation of the submission "DSRec: Mitigating Double Noise in LLM-Enhanced Sequential Recommendation via Dual-Space Embedding Adaptation".

## Environment

We use the following hardware and software environment:
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
By the way, we recommend you install the `tmux`.

## Dataset
1. Download the three datasets (**TripAdvisor**, **Yelp**, **Grocery & Gourmet Food**) from their official sources and place the raw files in the ```/PreProcess/raw_data```.

2. Execute the provided bash scripts in order to clean and format the data:
```sh
cd PreProcess

# Step 1: Filter cold-start users and items
bash scripts/trip_process_1.sh
bash scripts/yelp_process_1.sh
...

# Step 2: Generate data for LLM fine-tuning
bash scripts/trip_process_2.sh
bash scripts/yelp_process_2.sh
...
```

*Note*: For the Grocery dataset, run ```grocery_process_1.5.sh``` before process 2 to handle items with missing metadata.

3. Move the generated files to their respective directories for the next stages:
- Place ```item_str.jsonline``` into ```/GenSemantics/data/{dataset}/```
- Rename ```inter.txt``` to ```interaction.txt``` and place it in ```/data/{dataset}/```

## Representation Generation

### Semantic Embedding Generation (LLM-based)
0. *(Optional)* If NCCL-related issues occur during LLM fine-tuning, you may disable NCCL P2P and IB: 
```sh
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

1. Execute the provided scripts to fine-tune the LLM and ***generate item semantics***:
```sh
cd GenSemantics

bash scripts/semantic_trip.sh > ./results/llm_trip.log 2>&1
bash scripts/semantic_yelp.sh > ./results/llm_yelp.log 2>&1
bash scripts/semantic_grocery.sh > ./results/llm_grocery.log 2>&1
```

2. Run `/GenSemantics/results/convert.ipynb` to convert the generated item semantics from *.json* to *.pkl* format.

The output file `semantics_embeddings.pkl` should be placed in `/data/{dataset}`

### Collaborative Embedding Generation (ID-based)
1. Execute the provided bash scripts in order to ***generate the id-embeddings***:
```sh
bash scripts/general_trip.sh > ./results/id_trip.log 2>&1
bash scripts/general_yelp.sh > ./results/id_yelp.log 2>&1
bash scripts/general_grocery.sh > ./results/id_grocery.log 2>&1
```
The output are ```user_id_embeddings.pkl``` and ```item_id_embeddings.pkl```, working for the FACA.

## Adaptation
Now execute the provided bash to ***conduct dual-space adaptation***:
```sh
bash scripts/train_trip.sh > ./results/trip.log 2>&1
bash scripts/train_yelp.sh > ./results/yelp.log 2>&1
bash scripts/train_grocery.sh > ./results/grocery.log 2>&1
```

The checkpoint will be saved in the folder `/saved/`.

**Quick Start (Optional):**
For the TripAdvisor dataset, we provide the pre-generated semantic representations used in the experiments. 
This allows directly running the dual-space adaptation stage without executing the LLM-based preprocessing pipeline.
