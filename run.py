import argparse

from config import model_type
from models import u_net_clf
from models import u_net_clf_tpu


def run_experiment(args):
  model = args.model
  if model == model_type.ModelType.U_NET_CPU_GPU.name:
    u_net_clf.run_u_net(args.dataset)
  elif model == model_type.ModelType.U_NET_TPU.name:
    u_net_clf_tpu.run_u_net(args.dataset)
  else:
    raise NotImplementedError


if __name__ == '__main__':
  args_parser = argparse.ArgumentParser()

  args_parser.add_argument("--dataset",
                           required=True)

  args_parser.add_argument("--model",
                           required=True)

  args = args_parser.parse_args()
  run_experiment(args)
