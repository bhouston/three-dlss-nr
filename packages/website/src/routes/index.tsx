import { createFileRoute } from '@tanstack/react-router';

import { ExplainerPage } from '@/components/explainer/ExplainerPage';

export const Route = createFileRoute('/')({
  component: ExplainerPage,
});
