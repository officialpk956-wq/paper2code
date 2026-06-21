import { notFound } from "next/navigation";
import { LEARNING_TRACKS } from "@/data/learning-tracks";
import { TrackPageClient } from "@/components/track-page-client";

export default async function TrackPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const track = LEARNING_TRACKS.find((t) => t.slug === slug);

  if (!track) {
    notFound();
  }

  return <TrackPageClient track={track} />;
}
