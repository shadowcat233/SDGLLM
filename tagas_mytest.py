from TAGLAS.datasets import Cora, Pubmed
from TAGLAS.tasks import SubgraphTextNPTask
from TAGLAS.tasks.text_encoder import SentenceEncoder
dataset = Pubmed()
data = dataset[0]
print(data)
task = SubgraphTextNPTask(dataset)
encoder = SentenceEncoder('ST')
task.convert_text_to_embedding('ST', encoder)
# processed_edge_index, processed_node_map, processed_edge_map, mapping = \
#     task.__process_graph__(0, dataset[0].edge_index, dataset[0].node_map, dataset[0].edge_map)
# print(processed_edge_index)
# print(processed_edge_map)
# print(processed_node_map)
# print(mapping)
# print(task.edge_features.size(), task.label_features.size())

