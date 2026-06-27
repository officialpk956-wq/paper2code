def infer_mobilenet_defaults(schema):
    version = schema.get("version", "v2")
    if version == "v2":
        return {"multiplier": 1.0, "version": "v2",
                "stages": [
                    {"expand": 1, "out_ch": 16,  "n": 1, "stride": 1, "se": False, "hs": False},
                    {"expand": 6, "out_ch": 24,  "n": 2, "stride": 2, "se": False, "hs": False},
                    {"expand": 6, "out_ch": 32,  "n": 3, "stride": 2, "se": False, "hs": False},
                    {"expand": 6, "out_ch": 64,  "n": 4, "stride": 2, "se": False, "hs": False},
                    {"expand": 6, "out_ch": 96,  "n": 3, "stride": 1, "se": False, "hs": False},
                    {"expand": 6, "out_ch": 160, "n": 3, "stride": 2, "se": False, "hs": False},
                    {"expand": 6, "out_ch": 320, "n": 1, "stride": 1, "se": False, "hs": False},
                ]}
    # v3-small default
    return {"multiplier": 1.0, "version": "v3",
            "stages": [
                {"expand": 1, "out_ch": 16,  "n": 1, "stride": 2, "se": True,  "hs": False},
                {"expand": 4, "out_ch": 24,  "n": 1, "stride": 2, "se": False, "hs": False},
                {"expand": 3, "out_ch": 40,  "n": 2, "stride": 2, "se": True,  "hs": True},
                {"expand": 6, "out_ch": 48,  "n": 2, "stride": 1, "se": True,  "hs": True},
                {"expand": 6, "out_ch": 96,  "n": 3, "stride": 2, "se": True,  "hs": True},
            ]}
