# src/schema_rules_resnet.py

RESNET_BLOCK_RULES = {
    "basic": {
        "expansion": 1
    },
    "bottleneck": {
        "expansion": 4
    }
}

def infer_resnet_stage(
    prev_channels: int,
    stage_out_channels: int,
    block_type: str,
    stage_index: int
):
    expansion = RESNET_BLOCK_RULES[block_type]["expansion"]

    stride = 1 if stage_index == 0 else 2
    downsample = prev_channels != stage_out_channels * expansion

    return {
        "in_channels": prev_channels,
        "out_channels": stage_out_channels,
        "expansion": expansion,
        "stride": stride,
        "downsample": downsample
    }
