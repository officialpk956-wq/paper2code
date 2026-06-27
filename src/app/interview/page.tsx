import { INTERVIEW_QUESTIONS } from '@/data/interview-questions';
import { InterviewNotes } from '@/components/topic/InterviewNotes';

export default function InterviewPage() {
  // Map the new rich InterviewQuestion to the format expected by InterviewNotes
  const mappedQuestions = INTERVIEW_QUESTIONS.map(q => ({
    id: q.id,
    question: q.question,
    answer: q.sampleAnswer,
    tags: [q.category, q.difficulty, ...q.tags]
  }));

  return (
    <div className="page-container max-w-4xl mx-auto px-6 py-8">
      <div className="section-header mb-8">
        <h1 className="text-3xl font-black mb-2" style={{ color: 'var(--color-text-primary)' }}>Interview Prep</h1>
        <p style={{ color: 'var(--color-text-secondary)' }}>
          Ace your AI engineering interviews with guided practice and company-specific guides.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
        <div className="md:col-span-3">
          <InterviewNotes questions={mappedQuestions} />
        </div>
        <div className="space-y-4">
          <div className="p-4 rounded-xl" style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.07)' }}>
            <h3 className="text-sm font-bold mb-2 text-white">Categories</h3>
            <ul className="space-y-2 text-sm text-slate-400">
              <li>ML Fundamentals</li>
              <li>Deep Learning</li>
              <li>System Design</li>
              <li>Coding</li>
              <li>Behavioral</li>
            </ul>
          </div>
          
          <div className="p-4 rounded-xl" style={{ background: 'linear-gradient(135deg, rgba(124,58,237,0.1), rgba(6,182,212,0.1))', border: '1px solid rgba(139,92,246,0.2)' }}>
            <h3 className="text-sm font-bold mb-2 text-white">Need Mock Interviews?</h3>
            <p className="text-xs text-slate-300 mb-3">Try the Dojo simulator to practice answering these under time pressure.</p>
            <a href="/dojo" className="block text-center text-xs font-bold py-2 rounded-lg bg-indigo-500 text-white hover:bg-indigo-600 transition">Go to Dojo</a>
          </div>
        </div>
      </div>
    </div>
  );
}
