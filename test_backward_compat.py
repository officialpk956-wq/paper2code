from core.comparators import summarize_compute, compare_graphs, explain_architecture_comparison
from core.visualizer_resnet import build_resnet18_graph

g = build_resnet18_graph()
r = summarize_compute(g)
print(f"Original comparator working: {r['total_high_flops']} high-FLOPs nodes")
print("All imports successful - backward compatible!")
