import { BlockVizPage } from '@/components/block-viz/BlockVizPage';

export const metadata = {
  title: 'Block Explorer — Paper2Code',
  description: 'Explore real neural network block hierarchies with shapes and FLOPs from PyTorch.',
};

export default function Page() {
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <BlockVizPage />
    </div>
  );
}
