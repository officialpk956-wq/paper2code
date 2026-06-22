import { describe, it, expect } from 'vitest';
import { scoreAssessment, mapLevelToEntry } from '@/lib/assessment';
import { ASSESSMENT_QUESTIONS } from '@/data/assessment';

describe('Assessment Logic', () => {
  describe('scoreAssessment', () => {
    it('returns research-practitioner for all correct answers', () => {
      const answers: Record<string, string> = {};
      ASSESSMENT_QUESTIONS.forEach(q => {
        answers[q.id] = q.correctOptionId;
      });

      const result = scoreAssessment(answers);
      expect(result.level).toBe('research-practitioner');
      expect(result.skillProfile.foundations).toBe(1);
      expect(result.skillProfile.architectures).toBe(1);
      expect(result.skillProfile.systems).toBe(1);
      expect(result.skillProfile.implementation).toBe(1);
    });

    it('returns foundation-builder for all wrong answers', () => {
      const answers: Record<string, string> = {};
      ASSESSMENT_QUESTIONS.forEach(q => {
        // Pick an option that is definitely wrong
        answers[q.id] = q.options.find(opt => opt.id !== q.correctOptionId)?.id || '';
      });

      const result = scoreAssessment(answers);
      expect(result.level).toBe('foundation-builder');
      expect(result.skillProfile.foundations).toBe(0);
      expect(result.skillProfile.architectures).toBe(0);
      expect(result.skillProfile.systems).toBe(0);
      expect(result.skillProfile.implementation).toBe(0);
    });

    it('returns intermediate levels for mixed answers', () => {
      const answers: Record<string, string> = {};
      ASSESSMENT_QUESTIONS.forEach((q, index) => {
        if (index % 2 === 0) {
          answers[q.id] = q.correctOptionId; // half correct
        } else {
          answers[q.id] = q.options.find(opt => opt.id !== q.correctOptionId)?.id || '';
        }
      });

      const result = scoreAssessment(answers);
      expect(result.level).not.toBe('foundation-builder'); // should be better than nothing
      expect(result.level).not.toBe('research-practitioner'); // shouldn't be max
    });
  });

  describe('mapLevelToEntry', () => {
    it('maps level 1 to first phase of roadmap', () => {
      const result = mapLevelToEntry('foundation-builder', 'AI/ML Engineer');
      expect(result.roadmapId).toBe('ai-engineer');
      expect(result.startPhaseId).toBe('phase-1');
    });

    it('maps level 4 to last phase (or near last phase) depending on roadmap length', () => {
      const result = mapLevelToEntry('research-practitioner', 'Computer Vision');
      expect(result.roadmapId).toBe('cv-engineer');
      // CV Engineer only has 1 phase, so it should clamp to the only phase available
      expect(result.startPhaseId).toBe('phase-1');
      
      const resultAI = mapLevelToEntry('research-practitioner', 'AI/ML Engineer');
      // AI Engineer has 6 phases, index 3 corresponds to phase-4
      expect(resultAI.startPhaseId).toBe('phase-4');
    });

    it('defaults to ai-engineer if goal is unknown', () => {
      const result = mapLevelToEntry('system-builder', 'Unknown Goal');
      expect(result.roadmapId).toBe('ai-engineer');
      expect(result.startPhaseId).toBe('phase-3');
    });
  });
});
