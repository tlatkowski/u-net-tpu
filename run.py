import argparse


def run_experiment(dataset, model):
  pass


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument("dataset",
                             required=True)

    args_parser.add_argument("model",
                             required=True)

    args = args_parser.parse_args()
    run_experiment(args.dataset, args.model)