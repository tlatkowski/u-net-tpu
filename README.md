Under development

```bash
# TPU configuration
GCP_PROJECT=YOUR_GCP_PROJECT
TPU_ZONE=YOUR_TPU_ZONE
TPU_NAME=YOUR_TPU_NAME

# problem/model configuration
PROBLEM=MNIST


python u_net_clf_tpu.py \
--train_dir="~/tcl-research/git/u-net-tpu/places/train" \
--eval_dir="~/tcl-research/git/u-net-tpu/places/test" \
--problem=$PROBLEM \
--tpu_name=$TPU_NAME \
--tpu_zone=$TPU_ZONE \
--gcp_project=$GCP_PROJECT
```




python u_net_clf_tpu.py \
--train_dir="~/tcl-research/git/u-net-tpu/places/train" \
--eval_dir="~/tcl-research/git/u-net-tpu/places/test" \
--problem="MNIST" \
--tpu_name=tpu-vm \
--tpu_zone=us-central1-f \
--gcp_project=ml-research-tcl