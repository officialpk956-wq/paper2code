from src.diagram_base import create_graph

def draw_unet(schema):
    dot = create_graph("UNet Architecture")

    dot.node("input", "Input")

    prev = "input"
    encoders = []

    for i, stage in enumerate(schema["encoder"]):
        node = f"enc{i}"
        dot.node(node, f"Encoder {i}\nCh: {stage['out_channels']}")
        dot.edge(prev, node)
        encoders.append(node)
        prev = node

    dot.node("bottleneck", "Bottleneck")
    dot.edge(prev, "bottleneck")

    prev = "bottleneck"

    for i, stage in enumerate(schema["decoder"]):
        node = f"dec{i}"
        dot.node(node, f"Decoder {i}\nCh: {stage['out_channels']}")
        dot.edge(prev, node)
        dot.edge(encoders[-(i+1)], node, label="skip")
        prev = node

    dot.node("output", "Output")
    dot.edge(prev, "output")

    return dot
