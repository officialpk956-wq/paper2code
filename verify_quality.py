import sqlite3
import json

conn = sqlite3.connect("tensortonic_dev.db")
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT layer_name, flops_context, tensor_flow FROM paper_modules ORDER BY paper_id, order_index").fetchall()

print("=== QUALITY VERIFICATION ===")
print()

issues = []
for r in rows:
    name = r["layer_name"][:42]
    ctx_raw = r["flops_context"]
    tf_raw = r["tensor_flow"]

    ctx = {}
    if isinstance(ctx_raw, str):
        try:
            ctx = json.loads(ctx_raw)
        except Exception:
            ctx = {}
    elif isinstance(ctx_raw, dict):
        ctx = ctx_raw

    tf = {}
    if isinstance(tf_raw, str):
        try:
            tf = json.loads(tf_raw)
        except Exception:
            tf = {}
    elif isinstance(tf_raw, (list, dict)):
        tf = tf_raw

    # Confidence
    conf = ctx.get("confidence")
    if conf is None or (isinstance(conf, float) and (conf != conf)):  # NaN check
        conf_display = "NaN [FAIL]"
        issues.append(f"  NaN confidence: {name}")
    else:
        conf_display = f"{round(conf * 100)}%"

    # FLOPs
    mflops = ctx.get("real_flops_mflops", 0.0)
    if mflops is None or mflops == 0:
        mflops_display = "0 [WARN]"
    elif mflops >= 1000:
        mflops_display = f"{mflops/1000:.1f}G FLOPs"
    elif mflops >= 1:
        mflops_display = f"{mflops:.1f}M FLOPs"
    else:
        mflops_display = f"{mflops*1000:.0f}K FLOPs"

    # Tensor shapes
    in_shape = None
    out_shape = None
    if isinstance(tf, list) and len(tf) > 0:
        in_shape = tf[0].get("input_shape")
        out_shape = tf[-1].get("output_shape")
    elif isinstance(tf, dict):
        trace = tf.get("trace", [])
        if trace:
            in_shape = trace[0].get("input_shape")
            out_shape = trace[-1].get("output_shape")

    shape_display = f"{in_shape} -> {out_shape}" if (in_shape or out_shape) else "N/A"

    print(f"  {name:<43} conf={conf_display:<10}  {mflops_display:<18}  shapes={shape_display}")

print()
if issues:
    print(f"FAILURES FOUND ({len(issues)}):")
    for i in issues:
        print(i)
else:
    print("ALL PASS: No NaN confidence values found.")

conn.close()
