// lib/mapStore.ts

let mapImageData: string | null = null;
let mapParamsData: { MAP_SIZE: number[]; MAP_RESOLUTION: number } | null = null;

export const setMapImageData = (data: string) => {
  mapImageData = data;
};

export const getMapImageData = () => mapImageData;

export const setMapParamsData = (params: { MAP_SIZE: number[]; MAP_RESOLUTION: number }) => {
  mapParamsData = params;
};

export const getMapParamsData = () => mapParamsData;
