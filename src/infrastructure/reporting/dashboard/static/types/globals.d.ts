declare const Tabulator: any;

interface ApexGlobal {
  summary: any;
  scoreHistory: any;
  manifest: any;
  indicators: any;
  _cache: Record<string, any>;
  navigateTo: (page: string, params?: Record<string, string>) => void;
}

interface Window {
  APEX: ApexGlobal;
  Plotly: any;
  Tabulator: any;
}
