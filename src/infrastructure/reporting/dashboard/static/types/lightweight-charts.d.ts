/**
 * Lightweight Charts v5 — minimal type stubs for APEX dashboard.
 * Full types at: https://unpkg.com/lightweight-charts@5.0.0/dist/typings.d.ts
 */

declare module 'lightweight-charts' {
  export interface ChartOptions {
    width?: number;
    height?: number;
    autoSize?: boolean;
    layout?: {
      background?: { color: string };
      textColor?: string;
      fontSize?: number;
    };
    grid?: {
      vertLines?: { color: string };
      horzLines?: { color: string };
    };
    crosshair?: { mode?: number };
    timeScale?: {
      borderColor?: string;
      timeVisible?: boolean;
    };
    rightPriceScale?: {
      borderColor?: string;
    };
    [key: string]: any;
  }

  export interface IChartApi {
    addSeries(type: any, options?: any): ISeriesApi<any>;
    timeScale(): ITimeScaleApi;
    priceScale(id: string): { applyOptions(opts: any): void };
    applyOptions(opts: Partial<ChartOptions>): void;
    remove(): void;
  }

  export interface ITimeScaleApi {
    fitContent(): void;
    subscribeVisibleLogicalRangeChange(handler: (range: any) => void): void;
    getVisibleLogicalRange(): any;
    setVisibleLogicalRange(range: any): void;
  }

  export interface ISeriesApi<T> {
    setData(data: any[]): void;
    createPriceLine(options: any): any;
    applyOptions(opts: any): void;
  }

  export function createChart(container: HTMLElement, options?: ChartOptions): IChartApi;
  export function createSeriesMarkers(series: ISeriesApi<any>, markers: any[]): any;

  export const CandlestickSeries: any;
  export const LineSeries: any;
  export const HistogramSeries: any;
  export const AreaSeries: any;
}
