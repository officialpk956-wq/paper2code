from src.diagram_base import create_graph


def draw_resnet(schema):
    dot = create_graph("ResNet Architecture")

    dot.node("input", "Input")
    dot.node("stem", "Conv Stem")
    dot.edge("input", "stem")

    prev = "stem"

    for stage in schema["stages"]:
        stage_id = stage["name"]
        label = f"{stage['name']}\nBlocks: {stage['num_blocks']}\nChannels: {stage['out_channels']}"
        dot.node(stage_id, label)
        dot.edge(prev, stage_id)
        prev = stage_id

    dot.node("head", "AvgPool + FC")
    dot.edge(prev, "head")

    return dot
