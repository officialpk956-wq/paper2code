import torch
from core.schema_refiner_bert_gpt import refine_bert_gpt_schema
from core.bert_gpt_builder import BertGPTBuilder

def main():
    print("Testing BERT (is_causal=False)...")
    schema_bert = {"model_family": "bert_gpt", "variant": "bert"}
    refined_bert = refine_bert_gpt_schema(schema_bert)
    model_bert = BertGPTBuilder(refined_bert)
    input_ids = torch.randint(0, 100, (2, 32))
    out_bert = model_bert(input_ids)
    print(f"Output shape: {out_bert.shape}")
    print(f"Params: {sum(p.numel() for p in model_bert.parameters()):,}")

    print("\nTesting GPT (is_causal=True)...")
    schema_gpt = {"model_family": "bert_gpt", "variant": "gpt"}
    refined_gpt = refine_bert_gpt_schema(schema_gpt)
    model_gpt = BertGPTBuilder(refined_gpt)
    out_gpt = model_gpt(input_ids)
    print(f"Output shape: {out_gpt.shape}")
    print(f"Params: {sum(p.numel() for p in model_gpt.parameters()):,}")

if __name__ == "__main__":
    main()
