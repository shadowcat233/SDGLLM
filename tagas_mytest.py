from TAGLAS.datasets import Cora, Arxiv
from TAGLAS.tasks import SubgraphTextNPTask
from TAGLAS.tasks.text_encoder import SentenceEncoder
dataset = Arxiv()
import torch
task = SubgraphTextNPTask(dataset)

