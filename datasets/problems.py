import enum

from datasets import mnist, places


class Problem(enum.Enum):
  MNIST = 0,
  PLACES = 1


def get_problem(problem: Problem):
  if problem is Problem.MNIST:
    return mnist
  elif problem is Problem.PLACES:
    return places
  else:
    raise NotImplementedError


def all_problems():
  p = []
  for problem in Problem:
    p.append(problem.name)
  return p