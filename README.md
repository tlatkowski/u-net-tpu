**[Under development]**

### Run U-Net model on TPU Google Cloud Platform 

##### Create bucket for storing training checkpoints:

Execute the following commands to create a bucket on your TPU VM instance:
```bash
GCP_PROJECT=YOUR-GCP-PROJECT
MODEL_DIR=gs://YOUR-GOOGLE-CLOUD-BUCKET

gsutil mb -p $GCP_PROJECT $MODEL_DIR
```

##### Run U-Net model training on TPU flock:

```bash
# TPU configuration
GCP_PROJECT=YOUR-GCP-PROJECT
TPU_ZONE=YOUR-TPU-ZONE
TPU_NAME=YOUR-TPU-NAME
MODEL_DIR=gs://YOUR-GOOGLE-CLOUD-BUCKET

# problem/model configuration
PROBLEM=MNIST
TRAIN_DIR=PATH-TO-TRAIN-DIR
EVAL_DIR=PATH-TO-EVALUATION-DIR

python u_net_clf_tpu.py \
--train_dir=$TRAIN_DIR \
--eval_dir=$EVAL_DIR \
--problem=$PROBLEM \
--model_dir=$MODEL_DIR \
--tpu_name=$TPU_NAME \
--tpu_zone=$TPU_ZONE \
--gcp_project=$GCP_PROJECT
```




