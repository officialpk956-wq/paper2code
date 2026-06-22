'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ASSESSMENT_QUESTIONS, GOAL_OPTIONS, HOURS_OPTIONS } from '@/data/assessment';
import { scoreAssessment, mapLevelToEntry, saveProfile, loadProfile, hasProfile } from '@/lib/assessment';
import type { Profile } from '@/lib/assessment';

// total steps = 15 questions + 1 goal + 1 hours = 17 steps (index 0 to 16)
const TOTAL_STEPS = ASSESSMENT_QUESTIONS.length + 2;

export default function AssessmentPage() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [existingProfile, setExistingProfile] = useState<Profile | null>(null);
  const [showRetakePrompt, setShowRetakePrompt] = useState(false);

  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [goal, setGoal] = useState<string>('');
  const [hours, setHours] = useState<string>('');
  const [result, setResult] = useState<Profile | null>(null);

  useEffect(() => {
    setMounted(true);
    if (hasProfile()) {
      setExistingProfile(loadProfile());
      setShowRetakePrompt(true);
    }
  }, []);

  if (!mounted) return null;

  if (showRetakePrompt) {
    return (
      <div className="max-w-2xl mx-auto py-16 px-4">
        <h1 className="text-2xl font-bold text-[--color-text-primary] mb-4">You already have a skill profile</h1>
        <p className="text-[--color-text-secondary] mb-8">
          Current Level: <span className="font-semibold text-[--color-text-primary] capitalize">{existingProfile?.level.replace('-', ' ')}</span><br/>
          Current Goal: <span className="font-semibold text-[--color-text-primary]">{existingProfile?.goal}</span>
        </p>
        <div className="flex gap-4">
          <button
            onClick={() => router.push('/dashboard')}
            className="px-4 py-2 bg-[--accent-primary] text-white rounded hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[--accent-primary]"
          >
            Go to my dashboard
          </button>
          <button
            onClick={() => setShowRetakePrompt(false)}
            className="px-4 py-2 bg-[--bg-surface] text-[--color-text-primary] border border-[--color-border] rounded hover:bg-[--bg-panel] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[--accent-primary]"
          >
            Retake assessment
          </button>
        </div>
      </div>
    );
  }

  if (result) {
    return (
      <div className="max-w-2xl mx-auto py-16 px-4">
        <h1 className="text-3xl font-bold text-[--color-text-primary] mb-2">Assessment Complete</h1>
        <p className="text-lg text-[--color-text-secondary] mb-8">
          You're a <span className="font-bold text-[--accent-primary] capitalize">{result.level.replace('-', ' ')}</span> — your path: <span className="font-bold text-[--color-text-primary]">{result.goal}</span>
        </p>
        
        <div className="space-y-4 mb-8">
          <h2 className="text-lg font-semibold text-[--color-text-primary]">Skill Profile</h2>
          {Object.entries(result.skillProfile).map(([area, score]) => (
            <div key={area}>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-[--color-text-primary] capitalize">{area}</span>
                <span className="text-[--color-text-secondary]">{Math.round(score * 100)}%</span>
              </div>
              <div className="h-2 w-full bg-[--bg-surface] rounded overflow-hidden">
                <div 
                  className="h-full bg-[--accent-primary] transition-all duration-1000" 
                  style={{ width: `${score * 100}%` }}
                  role="progressbar"
                  aria-valuenow={score * 100}
                  aria-valuemin={0}
                  aria-valuemax={100}
                />
              </div>
            </div>
          ))}
        </div>

        <button
          onClick={() => router.push('/dashboard')}
          className="w-full py-3 bg-[--accent-primary] text-white rounded font-medium hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[--accent-primary]"
        >
          Go to my dashboard
        </button>
      </div>
    );
  }

  const isQuestionStep = step < ASSESSMENT_QUESTIONS.length;
  const isGoalStep = step === ASSESSMENT_QUESTIONS.length;
  const isHoursStep = step === ASSESSMENT_QUESTIONS.length + 1;

  const handleNext = () => {
    if (isHoursStep) {
      // Finish
      const { skillProfile, level } = scoreAssessment(answers);
      const { roadmapId, startPhaseId } = mapLevelToEntry(level, goal);
      const newProfile: Profile = {
        level,
        goal,
        roadmapId,
        startPhaseId,
        hoursPerWeek: hours,
        skillProfile,
        completedAt: Date.now()
      };
      saveProfile(newProfile);
      setResult(newProfile);
    } else {
      setStep(s => s + 1);
    }
  };

  const handleBack = () => {
    setStep(s => Math.max(0, s - 1));
  };

  const canProceed = () => {
    if (isQuestionStep) {
      return !!answers[ASSESSMENT_QUESTIONS[step].id];
    }
    if (isGoalStep) return !!goal;
    if (isHoursStep) return !!hours;
    return false;
  };

  const progressPercent = ((step + 1) / TOTAL_STEPS) * 100;

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-2xl mx-auto py-16 px-4 min-h-full">
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between text-xs text-[--color-text-tertiary] mb-2">
            <span>Step {step + 1} of {TOTAL_STEPS}</span>
            <span>{Math.round(progressPercent)}%</span>
          </div>
          <div className="h-1.5 w-full bg-[--bg-surface] rounded overflow-hidden">
            <div 
              className="h-full bg-[--accent-primary] transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
              role="progressbar"
              aria-valuenow={progressPercent}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          </div>
        </div>

        {/* Content */}
        <div className="bg-[--bg-panel] border border-[--color-border] rounded-xl p-6 mb-6">
          {isQuestionStep && (
            <fieldset>
              <legend className="text-xl font-semibold text-[--color-text-primary] mb-6">
                {ASSESSMENT_QUESTIONS[step].question}
              </legend>
              <div className="space-y-3">
                {ASSESSMENT_QUESTIONS[step].options.map(opt => (
                  <label 
                    key={opt.id}
                    className={`block border rounded-lg p-4 cursor-pointer transition-colors focus-within:ring-2 focus-within:ring-[--accent-primary] ${
                      answers[ASSESSMENT_QUESTIONS[step].id] === opt.id 
                        ? 'border-[--accent-primary] bg-[--bg-surface]' 
                        : 'border-[--color-border] hover:border-[--color-text-tertiary]'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <input 
                        type="radio" 
                        name={ASSESSMENT_QUESTIONS[step].id} 
                        value={opt.id}
                        checked={answers[ASSESSMENT_QUESTIONS[step].id] === opt.id}
                        onChange={(e) => setAnswers({...answers, [ASSESSMENT_QUESTIONS[step].id]: e.target.value})}
                        className="w-4 h-4 text-[--accent-primary] focus-visible:outline-none"
                        aria-label={opt.text}
                      />
                      <span className="text-[--color-text-primary] text-sm">{opt.text}</span>
                    </div>
                  </label>
                ))}
              </div>
            </fieldset>
          )}

          {isGoalStep && (
            <fieldset>
              <legend className="text-xl font-semibold text-[--color-text-primary] mb-6">
                What is your primary learning goal?
              </legend>
              <div className="space-y-3">
                {Object.keys(GOAL_OPTIONS).map(g => (
                  <label 
                    key={g}
                    className={`block border rounded-lg p-4 cursor-pointer transition-colors focus-within:ring-2 focus-within:ring-[--accent-primary] ${
                      goal === g 
                        ? 'border-[--accent-primary] bg-[--bg-surface]' 
                        : 'border-[--color-border] hover:border-[--color-text-tertiary]'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <input 
                        type="radio" 
                        name="goal" 
                        value={g}
                        checked={goal === g}
                        onChange={(e) => setGoal(e.target.value)}
                        className="w-4 h-4 text-[--accent-primary] focus-visible:outline-none"
                      />
                      <span className="text-[--color-text-primary] text-sm">{g}</span>
                    </div>
                  </label>
                ))}
              </div>
            </fieldset>
          )}

          {isHoursStep && (
            <fieldset>
              <legend className="text-xl font-semibold text-[--color-text-primary] mb-6">
                How much time can you commit per week?
              </legend>
              <div className="space-y-3">
                {HOURS_OPTIONS.map(h => (
                  <label 
                    key={h}
                    className={`block border rounded-lg p-4 cursor-pointer transition-colors focus-within:ring-2 focus-within:ring-[--accent-primary] ${
                      hours === h 
                        ? 'border-[--accent-primary] bg-[--bg-surface]' 
                        : 'border-[--color-border] hover:border-[--color-text-tertiary]'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <input 
                        type="radio" 
                        name="hours" 
                        value={h}
                        checked={hours === h}
                        onChange={(e) => setHours(e.target.value)}
                        className="w-4 h-4 text-[--accent-primary] focus-visible:outline-none"
                      />
                      <span className="text-[--color-text-primary] text-sm">{h}</span>
                    </div>
                  </label>
                ))}
              </div>
            </fieldset>
          )}
        </div>

        {/* Navigation */}
        <div className="flex justify-between">
          <button
            onClick={handleBack}
            disabled={step === 0}
            className="px-4 py-2 text-[--color-text-secondary] disabled:opacity-50 hover:text-[--color-text-primary] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[--accent-primary] rounded"
          >
            Back
          </button>
          <button
            onClick={handleNext}
            disabled={!canProceed()}
            className="px-6 py-2 bg-[--accent-primary] text-white rounded font-medium disabled:opacity-50 hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[--accent-primary]"
          >
            {isHoursStep ? 'Finish' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
}
