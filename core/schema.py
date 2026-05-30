# src/schema.py

MODEL_SCHEMA = {
    "model_name": None,
    "task": None,  # classification, segmentation, seq2seq, detection, etc.

    "components": [
        # Example component
        # {
        #   "type": "conv | attention | embedding | encoder | decoder | ffn | other",
        #   "details": {}
        # }
    ],

    "loss": None,
    "optimizer": None,

    "training_details": {
        "batch_size": None,
        "epochs": None,
        "learning_rate": None
    },

    "notes": None
}
