'use client';

import { motion } from 'framer-motion';
import StaggerList from '@/components/StaggerList';

interface SectionGridProps {
  sections: string[];
}

export function SectionGrid({ sections }: SectionGridProps) {
  return (
    <StaggerList>
      {sections.map((section) => (
        <motion.div
          key={section}
          className="bg-[#111111] border border-[#262626] border-dashed rounded-xl p-8 text-center my-3"
        >
          <div className="text-[12px] font-semibold text-[#525252] uppercase tracking-wider mb-2">
            {section}
          </div>
          <p className="text-[13px] text-[#A3A3A3] italic">
            Content in production — structure follows the Paper2Code chapter template
          </p>
        </motion.div>
      ))}
    </StaggerList>
  );
}

export default SectionGrid;
