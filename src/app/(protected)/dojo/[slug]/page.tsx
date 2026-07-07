import { getMdxContent } from '@/lib/mdx';
import MdxRenderer from '@/components/MdxRenderer';
import DojoEditor from './DojoEditor';

export default async function Page({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const mdxSource = getMdxContent('problems', slug);

  return (
    <DojoEditor>
      {mdxSource ? <MdxRenderer source={mdxSource} /> : null}
    </DojoEditor>
  );
}
