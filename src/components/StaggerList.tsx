'use client';

import React from 'react';
import { motion } from 'framer-motion';

interface StaggerListProps {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}

const containerVariants = (delay: number) => ({
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.06,
      delayChildren: delay,
    },
  },
});

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.35, ease: 'easeOut' as const },
  },
};

export function StaggerList({ children, className, delay = 0 }: StaggerListProps) {
  return (
    <motion.div
      className={className}
      variants={containerVariants(delay)}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-40px' }}
    >
      {React.Children.map(children, (child, index) => (
        <motion.div key={index} variants={itemVariants}>
          {child}
        </motion.div>
      ))}
    </motion.div>
  );
}

export default StaggerList;
