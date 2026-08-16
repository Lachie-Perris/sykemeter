const FORECAST = "final_surf_forecast.svg";
const fallback = { height:"0.23 m", rating:"Poor", swell:"6.0 s from SE", wind:"18.1 kt from SE", tide:"1.10 m ↓ Falling", updated:"Current conditions", run:"GFS Wave forecast" };

function textValues(svg) {
  const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
  return [...doc.querySelectorAll("text")].map(node => node.textContent.trim()).filter(Boolean);
}

function setField(name, value) {
  document.querySelectorAll(`[data-field="${name}"]`).forEach(el => { el.textContent = value; });
}

async function refreshForecast() {
  const cacheKey = Date.now();
  document.querySelector("#forecast-image").src = `${FORECAST}?v=${cacheKey}`;
  try {
    const response = await fetch(`${FORECAST}?v=${cacheKey}`, { cache:"no-store" });
    if (!response.ok) throw new Error("Forecast unavailable");
    const values = textValues(await response.text());
    const after = label => { const i = values.indexOf(label); return i >= 0 ? values[i + 1] : ""; };
    const conditions = {
      height: after("Tide-adjusted Hs") || fallback.height,
      rating: after("Rating") || fallback.rating,
      swell: after("Swell") || fallback.swell,
      wind: after("Wind") || fallback.wind,
      tide: after("Tide") || fallback.tide,
      updated: (values.find(v => v.startsWith("Current conditions")) || fallback.updated).replace("Current conditions · ", ""),
      run: values.find(v => v.startsWith("GFS Wave run")) || fallback.run
    };
    Object.entries(conditions).forEach(([key,value]) => setField(key,value));
    document.querySelector("#rating-dot").className = `rating-dot ${conditions.rating.toLowerCase().includes("poor") ? "poor" : "fair"}`;
  } catch {
    Object.entries(fallback).forEach(([key,value]) => setField(key,value));
  }
}

refreshForecast();
