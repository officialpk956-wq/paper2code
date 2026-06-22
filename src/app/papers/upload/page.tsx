import { PaperUploadWorkspace } from '@/components/paper-upload/PaperUploadWorkspace';

export const metadata = {
  title: 'Upload Paper — Paper2Code',
  description: 'Upload a research paper PDF and generate a persisted Paper2Code workspace.',
};

export default function PaperUploadPage() {
  return <PaperUploadWorkspace />;
}
