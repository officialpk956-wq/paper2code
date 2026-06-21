'use client';

interface PaperContentProps {
  section?: string;
}

const sectionContent: Record<string, { title: string; content: string }> = {
  abstract: {
    title: 'Abstract',
    content: `Transformers have become the dominant architecture in modern deep learning,
particularly for sequence-to-sequence tasks. In this work, we propose a novel approach to
improving attention mechanisms through efficient computation and better generalization.
Our experimental results demonstrate significant improvements on multiple benchmarks,
achieving state-of-the-art performance while reducing computational complexity.
We provide extensive analysis of the attention patterns learned by our model and
demonstrate that these patterns align with linguistic intuitions.`,
  },
  intro: {
    title: 'Introduction',
    content: `The transformer architecture, introduced by Vaswani et al. (2017), has revolutionized
the field of natural language processing and beyond. The key innovation—multi-head self-attention—
allows models to attend to different positions in a sequence in parallel.

However, the quadratic time and space complexity of standard attention limits its application
to long sequences. Additionally, attention patterns often exhibit redundancy across different
heads and positions. Our work addresses these limitations through:

1. Efficient attention computation reducing complexity to O(n log n)
2. Sparse attention patterns focusing on local and global interactions
3. Improved training procedures for better convergence
4. Comprehensive analysis of learned attention patterns

We validate our approach on various tasks including machine translation, language modeling,
and question answering, demonstrating consistent improvements.`,
  },
  method: {
    title: 'Methodology',
    content: `We propose a sparse attention mechanism that selectively computes attention
between relevant positions rather than all position pairs.

The core idea is to decompose attention into:
• Local attention: each token attends to nearby positions
• Strided attention: sparse sampling of the sequence
• Learned sparse patterns: data-dependent sparsity

Mathematical formulation:

Attention(Q, K, V) = softmax(QK^T / √d_k) V

where sparse version uses a mask M:

SparseAttention(Q, K, V) = softmax((QK^T ⊙ M) / √d_k) V

The sparsity pattern M is learned during training, allowing the model to discover
efficient attention structures. We implement this using block-sparse matrix operations
for efficient GPU computation.`,
  },
  experiments: {
    title: 'Experiments',
    content: `We evaluate our approach on three main tasks:

1. Machine Translation (WMT14 En-De)
   - Baseline: 28.4 BLEU
   - Our method: 29.2 BLEU (+0.8)
   - Training time: 40% reduction

2. Language Modeling (WikiText-103)
   - Baseline: 18.3 perplexity
   - Our method: 17.6 perplexity
   - Sequence length: 4096 tokens (vs 1024 baseline)

3. QA Tasks (SQuAD v2)
   - Baseline: 86.2 F1
   - Our method: 87.1 F1

All experiments use the same hyperparameters except for sparse pattern learning rate.
We report results over 5 runs with different random seeds.`,
  },
  results: {
    title: 'Results & Discussion',
    content: `Our experiments demonstrate that sparse attention patterns can match or exceed
the performance of dense attention while significantly reducing computational costs.

Key findings:
• Sparse patterns are interpretable and align with linguistic structure
• Local attention captures syntax, long-range patterns capture semantics
• The learned sparsity differs by task, confirming data-dependent structure
• Efficiency gains scale with sequence length (up to 8x speedup at 4096 tokens)

Attention visualization reveals:
- Attention to punctuation marks for structural information
- Long-range dependencies for pronoun resolution
- Positional dependencies for word sense disambiguation

Limitations and future work:
- Current approach best suited for tasks with clear structural patterns
- Potential improvements through hybrid sparse-dense attention
- Application to vision transformers as future research direction`,
  },
};

export function PaperContent({ section = 'abstract' }: PaperContentProps) {
  const content = sectionContent[section] || sectionContent.abstract;

  return (
    <div className="flex flex-col h-full bg-[--bg-body]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[--color-border] bg-[--bg-panel]">
        <h2 className="text-xl font-bold text-[--color-text-primary]">{content.title}</h2>
        <p className="text-sm text-[--color-text-secondary] mt-1">
          Attention Is All You Need (2017)
        </p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-3xl mx-auto">
          <div className="prose prose-invert max-w-none">
            <div className="text-[15px] leading-relaxed text-[--color-text-primary] space-y-4">
              {content.content.split('\n\n').map((paragraph, idx) => (
                <p key={idx} className="whitespace-pre-wrap">
                  {paragraph}
                </p>
              ))}
            </div>
          </div>

          {/* Highlight boxes */}
          <div className="mt-8 p-4 bg-[--accent-primary]/10 border border-[--accent-primary]/20 rounded-lg">
            <div className="text-xs font-semibold text-[--accent-primary] mb-2">KEY INSIGHT</div>
            <p className="text-sm text-[--color-text-secondary] leading-relaxed">
              Multi-head self-attention with sparse patterns achieves state-of-the-art results
              while reducing computational complexity, enabling efficient processing of longer sequences.
            </p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="px-6 py-4 border-t border-[--color-border] bg-[--bg-panel] flex items-center justify-between">
        <button className="text-xs font-medium text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          ← Previous Section
        </button>
        <div className="text-xs text-[--color-text-tertiary]">Estimated time: 5 min</div>
        <button className="text-xs font-medium text-[--color-text-secondary] hover:text-[--color-text-primary] transition-colors">
          Next Section →
        </button>
      </div>
    </div>
  );
}
