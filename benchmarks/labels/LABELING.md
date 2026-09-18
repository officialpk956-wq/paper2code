# Labeling procedure

Labels are ground truth. A label that is wrong makes every measurement wrong
in a way that looks like an extractor result. Ten papers taught the failure
modes below; follow the procedure exactly so that paper 25 is labelled the
same way as paper 1.

## The one rule

**Label only what the paper explicitly states about its own model.**
Not what you know the architecture to contain. Not what the model predicted.
Not what a related-work paragraph describes.

## Procedure for a new paper

1. **Create the label** with `paper_id`, `source` (`arxiv:<id>`), `family`,
   and an empty `expected.layer_types`. If the paper describes several model
   sizes (ViT-Base/Large/Huge, BERT-Base/Large), set `variant` to the one
   being labelled. `variant` names a configuration; it never carries numbers.

2. **Run the neutral scan** -- before looking at any model output:

       python -m benchmarks.label_scan <paper_id>

   It prints, for every canonical type not yet in the label, the first
   sentence in the paper mentioning it. It reads exactly what the extractor
   reads (same PDF, same text extraction). Confirm the scan reports
   `scanned 1 of 1`; a paper that cannot be read is an error, never a skip.

3. **Read every hit.** A regex match is a prompt to read, not a decision.
   Reject the hit when the sentence is:
   - the paper **rejecting** the component -- "we do not use dropout",
     "does not have any fully connected layers", "rather than the standard
     relu", "replaces maxpooling with strided convolutions";
   - **related work** or a baseline -- "ResNet has five stages, and batch
     normalization...", "replace the Batch Normalization layers" (about the
     comparison CNN, not the model);
   - **preprocessing or evaluation** -- "center-crops of training examples",
     "single-crop, single-model", a de-duplication autoencoder;
   - a **reference title** -- "Sigmoid-weighted linear units" in the
     bibliography;
   - an **internal of a component already labelled**, unless the paper names
     it as a layer of its own -- softmax inside attention, concatenation of
     heads inside MSA.

   Accept the hit when the paper states it as part of the model being
   labelled, including in its architecture table or Figure 1. A figure label
   counts (`[CLS]` in BERT's Figure 1); a garbled column-merged table row
   counts if the component is legible ("Conv1x1 & Pooling & FC").

4. **Record the sentence.** Every accepted type gets its quote in `notes`:

       "dropout: \"We use a dropout probability of 0.1 on all layers\""

   If you cannot quote it, you cannot label it.

5. **Hyperparameters** (`key_hyperparams`): only values the paper states for
   the labelled variant, from a table row or a sentence. Never from an
   ablation or a scaling experiment. Check that each value appears in the
   paper's text as a token (`768`, not "seven hundred sixty-eight") -- the
   verification guard rejects replacement values it cannot find.

6. **Vocabulary.** If the paper names a component the canonical vocabulary
   lacks (`mbconv` appeared at paper 5), add it to `CANONICAL_TYPES` and
   `_SYNONYM_MAP` in `core/rag/normalizer.py` with the paper's spellings.
   Synonym-only changes do not invalidate staged extractions; scoring
   re-applies the table.

7. **Validate**: `python -m pytest tests/test_benchmark_harness.py -q` loads
   every label and rejects unknown types, duplicates and bad schema.

## What went wrong before, so it does not again

- **Model-informed selection.** The first curation pass checked only the
  types the model had predicted. Every addition was genuine, but types the
  papers state and the model *misses* were never sought, so precision was
  inflated. The neutral scan exists so this cannot recur: it reads the
  paper, not the prediction.
- **A silently skipped paper.** The first neutral scan `continue`d past a
  paper whose cached PDF had been deleted, printed nine sections where ten
  were expected, and nobody counted. The scan now fails on a missing paper
  and prints `scanned N of M`.
- **A label expecting an unstated fact.** `densenet121` expected
  `num_classes: 1000`; the paper's only `1000` is "> 1000 layers". Step 5
  would have caught it.
- **Scoring spelling.** `add_norm` for the Transformer's "Add & Norm"
  scored as a false positive *and* a missed `layernorm` until scoring
  applied the synonym table. Step 6 keeps the vocabulary honest.

## What a label is not

It is not complete. A paper states what it states; U-Net's label has six
types because U-Net's paper names six. Recall is measured against what is
stated, and a model that finds components the paper does not name is not
rewarded for it. That is deliberate: the benchmark measures reading, not
prior knowledge.
