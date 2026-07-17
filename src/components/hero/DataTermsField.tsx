// Faint, drifting data-science terms scattered across the fixed background
// layer — ambient technical texture behind the neural-field canvas. Static
// positions (no randomness) so this stays server-renderable with zero
// hydration risk; motion is pure CSS (see .p2c-term in globals.css).
const TERMS: { text: string; top: string; left: string; size: number; delay: number; duration: number }[] = [
  { text: 'gradient descent', top: '10%', left: '6%',  size: 12, delay: 0,  duration: 14 },
  { text: 'backpropagation',  top: '18%', left: '82%', size: 11, delay: 2,  duration: 16 },
  { text: 'attention',        top: '30%', left: '90%', size: 13, delay: 1,  duration: 12 },
  { text: 'transformer',      top: '6%',  left: '40%', size: 12, delay: 3,  duration: 15 },
  { text: 'embedding',        top: '46%', left: '4%',  size: 11, delay: 1.5, duration: 13 },
  { text: 'tensor',           top: '60%', left: '88%', size: 13, delay: 0.5, duration: 17 },
  { text: 'softmax',          top: '72%', left: '8%',  size: 11, delay: 2.5, duration: 14 },
  { text: 'cross-entropy',    top: '85%', left: '78%', size: 11, delay: 1,  duration: 16 },
  { text: 'overfitting',      top: '38%', left: '55%', size: 10, delay: 3.5, duration: 12 },
  { text: 'latent space',     top: '92%', left: '30%', size: 12, delay: 0.8, duration: 15 },
  { text: 'fine-tuning',      top: '54%', left: '22%', size: 10, delay: 2.2, duration: 13 },
  { text: 'ReLU',             top: '15%', left: '65%', size: 12, delay: 1.8, duration: 14 },
  { text: 'eigenvector',      top: '68%', left: '48%', size: 10, delay: 0.3, duration: 16 },
  { text: 'hyperparameter',   top: '80%', left: '55%', size: 10, delay: 2.8, duration: 15 },
  { text: 'inference',        top: '24%', left: '14%', size: 11, delay: 1.2, duration: 13 },
  { text: 'autograd',         top: '96%', left: '68%', size: 10, delay: 3.2, duration: 14 },
];

export function DataTermsField() {
  return (
    <div className="absolute inset-0 overflow-hidden" aria-hidden="true">
      {TERMS.map((t) => (
        <span
          key={t.text}
          className="p2c-term absolute whitespace-nowrap [font-family:var(--font-mono)] uppercase tracking-[0.12em] text-[#66F5FF]"
          style={{
            top: t.top,
            left: t.left,
            fontSize: t.size,
            animationDelay: `${t.delay}s`,
            animationDuration: `${t.duration}s`,
          }}
        >
          {t.text}
        </span>
      ))}
    </div>
  );
}

export default DataTermsField;
