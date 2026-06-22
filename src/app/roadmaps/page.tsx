import { getAllSlugs } from "@/lib/content/loader";
import { RoadmapsClientPage } from "./client-page";

export default function RoadmapsPage() {
  const paperSlugs = getAllSlugs("paper");
  const architectureSlugs = getAllSlugs("architecture");
  const systemDesignSlugs = getAllSlugs("system-design");
  const problemSlugs = getAllSlugs("problem");
  const implementationSlugs = getAllSlugs("implementation");

  const validSlugs = {
    paper: paperSlugs,
    architecture: architectureSlugs,
    "system-design": systemDesignSlugs,
    problem: problemSlugs,
    implementation: implementationSlugs,
  };

  return <RoadmapsClientPage validSlugs={validSlugs} />;
}
