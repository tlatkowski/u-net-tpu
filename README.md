Under development

To run U-Net model on TPU Google Cloud Platform 
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




