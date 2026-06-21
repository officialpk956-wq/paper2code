import { PROBLEMS } from '@/data/problems';
import { DojoProblemList } from '@/components/dojo/DojoProblemList';

export const metadata = {
  title: 'DS Coding Dojo — Paper2Code',
  description: 'LeetCode-style Data Science & ML coding practice with instant Python test execution.',
};

export default function DojoPage() {
  return <DojoProblemList problems={PROBLEMS} />;
}
