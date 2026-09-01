import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

# The regex matches the entire block of TAB 6
pattern = re.compile(r"          \{\/\* TAB 6: CYCLONE & EDDY FORECASTING \*\/\}\n          \{activeTab === \"forecasting\" && \(\n            <div className=\"space-y-6\">\n              <div className=\"bg-surface-white shadow-sm border border-glass-border border border-glass-border p-6  space-y-4\">\n                <div>\n                  <h2 className=\"font-headline-md text-headline-md text-on-surface flex items-center gap-2\">\n                    <TrendingUp className=\"w-5 h-5 text-primary\" \/>\n                    Mesoscale Eddy & Cyclone Latent Trajectory Forecaster\n                  <\/h2>\n                  <p className=\"font-body-sm text-body-sm text-text-muted\">\n                    Recurrent LSTM forecasting 1-day ahead ocean eddy migration in 256-D latent space\n                  <\/p>\n                <\/div>\n\n                <div className=\" overflow-hidden border border-glass-border bg-background text-on-surface\">\n                  <img\n                    src=\"\/assets\/cyclone_eddy_forecast_track.png\"\n                    alt=\"Cyclone & Eddy Forecast Track\"\n                    className=\"w-full object-cover\"\n                  \/>\n                <\/div>\n              <\/div>\n            <\/div>\n          \)\}", re.DOTALL)


replacement = r"""          {/* TAB 6: CYCLONE & EDDY FORECASTING */}
          {activeTab === "forecasting" && (
            (() => {
              // Mathematical calculation of the live trajectory based on AI UOHC fuel
              let cycloneTrack: any = null;
              if (inferResults) {
                const startLon = lon;
                const startLat = lat;
                const tchp = Number(inferResults.derived_physical_products?.tchp_kj_cm2 ?? 50);
                const I0 = Math.max(1, seedStormCategory); // Assume at least Cat 1 for track
                let currentLon = startLon;
                let currentLat = startLat;
                let intensity = I0;
                
                const coordinates = [[currentLon, currentLat]];
                
                for (let t = 1; t <= 168; t += 6) {
                   // Kinetic Decay Model tied directly to AI UOHC fuel
                   intensity = I0 * Math.exp(-0.25 * t / Math.max(1, (tchp / 15)));
                   if (intensity < 0.2) break; // Storm dies
                   
                   // NW steering flow + Coriolis
                   const driftLat = 0.05 + (currentLat * 0.002);
                   const driftLon = -0.05;
                   
                   currentLon += driftLon * intensity;
                   currentLat += driftLat * intensity;
                   coordinates.push([currentLon, currentLat]);
                }
                
                if (coordinates.length > 1) {
                  cycloneTrack = {
                    type: "Feature",
                    geometry: { type: "LineString", coordinates }
                  };
                }
              }

              return (
                <div className="space-y-6">
                  {/* Forecaster Header */}
                  <div className="bg-surface-white border border-glass-border p-6 shadow-md rounded-xl">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-xl font-bold text-on-surface flex items-center gap-2.5">
                        <TrendingUp className="w-5 h-5 text-rose-500" /> AI-Driven Trajectory Engine: Kinetic Decay Model
                      </h4>
                      <span className="bg-rose-500/10 text-rose-500 px-3 py-1 text-xs font-bold rounded-lg border border-rose-500/20">LIVE MAP INJECTION</span>
                    </div>
                    <p className="text-sm text-text-muted">
                      The cyclone path drawn on the map is calculated live. It uses your injected Seed Storm Category as initial inertia, and the AI's deep thermal fuel prediction (UOHC) at the origin coordinate as the battery life. <strong>If UOHC is high, the storm reaches the coast. If UOHC is low, it dies in the ocean.</strong> Change the SST/SSH inputs on the right to watch the downstream path manipulate!
                    </p>
                  </div>

                  {/* Main Grid: Map on Left, Inputs on Right */}
                  <div className="flex flex-col lg:flex-row gap-6">
                    {/* Left: Map */}
                    <div className="lg:w-3/4 relative bg-surface-container border border-glass-border min-h-[600px] overflow-hidden flex flex-col rounded-xl">
                      <Map
                        initialViewState={{
                          longitude: 85,
                          latitude: 15,
                          zoom: 4,
                          pitch: 0,
                          bearing: 0
                        }}
                        maxBounds={[[40, 0], [110, 35]] as any}
                        mapStyle={{
                          version: 8,
                          sources: {
                            esri: {
                              type: "raster",
                              tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
                              tileSize: 256,
                              attribution: "Esri"
                            }
                          },
                          layers: [
                            {
                              id: "esri-layer",
                              type: "raster",
                              source: "esri",
                              minzoom: 0,
                              maxzoom: 19
                            }
                          ]
                        }}
                        onClick={(e) => {
                          setLon(Math.min(Math.max(Number(e.lngLat.lng.toFixed(2)), 45), 105));
                          setLat(Math.min(Math.max(Number(e.lngLat.lat.toFixed(2)), 5), 30));
                        }}
                        cursor="crosshair"
                      >
                        <Source id="graticule" type="geojson" data={GRATICULE_GEOJSON}>
                          <Layer 
                            id="graticule-line" 
                            type="line" 
                            paint={{
                              "line-color": "#06b6d4",
                              "line-opacity": 0.4,
                              "line-width": 1,
                              "line-dasharray": [3, 3]
                            }} 
                          />
                        </Source>
                        
                        {cycloneTrack && (
                          <Source id="cyclone-track" type="geojson" data={cycloneTrack}>
                            <Layer 
                              id="cyclone-track-line" 
                              type="line" 
                              paint={{
                                "line-color": "#ef4444",
                                "line-width": 5,
                                "line-opacity": 0.9,
                                "line-blur": 2
                              }} 
                            />
                          </Source>
                        )}
                        
                        <NavigationControl position="bottom-right" />
                        
                        {!isNaN(lat) && !isNaN(lon) && (
                          <Marker 
                            longitude={Math.min(Math.max(lon, 45), 105)} 
                            latitude={Math.min(Math.max(lat, 5), 30)} 
                            anchor="center"
                            draggable={true}
                            onDrag={(e) => {
                              setLon(Math.min(Math.max(Number(e.lngLat.lng.toFixed(2)), 45), 105));
                              setLat(Math.min(Math.max(Number(e.lngLat.lat.toFixed(2)), 5), 30));
                            }}
                          >
                            <div className="text-rose-500 cursor-grab active:cursor-grabbing transition-transform hover:scale-125 drop-shadow-[0_4px_10px_rgba(244,63,94,0.8)]">
                              <Crosshair className="w-10 h-10 stroke-[2.5]" />
                            </div>
                          </Marker>
                        )}
                      </Map>
                      
                      {/* Seed Storm Category Controller Overlay */}
                      <div className="absolute bottom-6 left-6 right-16 bg-background/95 backdrop-blur-md p-4 border-2 border-rose-500 shadow-xl rounded-xl z-10 flex flex-col gap-2">
                        <div className="flex justify-between items-center mb-1">
                          <span className="font-bold text-sm text-on-surface">Inject Seed Storm Category:</span>
                          <span className="font-mono text-rose-500 font-bold bg-rose-500/10 px-2 rounded">Cat {seedStormCategory}</span>
                        </div>
                        <input 
                          type="range" 
                          min="1" max="5" step="1" 
                          value={seedStormCategory} 
                          onChange={(e) => setSeedStormCategory(parseInt(e.target.value))}
                          className="w-full accent-rose-500 cursor-pointer" 
                        />
                        <div className="flex justify-between text-xs text-text-muted font-bold px-1">
                          <span>Cat 1 (Depression)</span>
                          <span>Cat 3 (Severe)</span>
                          <span>Cat 5 (Super Cyclone)</span>
                        </div>
                      </div>
                    </div>

                    {/* Right: The 7-Inputs so User is never disconnected */}
                    <div className="lg:w-1/4 flex flex-col gap-2 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
                      <div className="bg-surface-white border border-rose-500/40 shadow-sm p-3 mb-2 rounded-xl text-center">
                        <p className="text-xs font-bold text-rose-500">Live Prediction Coordinates</p>
                        <p className="font-mono text-sm">{lat.toFixed(2)}°N, {lon.toFixed(2)}°E</p>
                      </div>
                      
                      <button 
                        onClick={handleRunInference}
                        disabled={isLoading}
                        className={`w-full py-4 rounded-xl flex items-center justify-center gap-2 font-button-lg text-button-lg shadow-sm border border-glass-border font-bold mb-4 ${isLoading ? "bg-surface-container-high text-on-surface-variant border-surface-container-highest cursor-not-allowed" : "bg-primary text-on-primary hover:bg-primary-hover active:bg-primary-active border-primary hover:shadow-md transition-all"}`}
                      >
                        {isLoading ? (
                          <>
                            <RotateCcw className="w-5 h-5 animate-spin" />
                            <span>Computing 3D Subsurface...</span>
                          </>
                        ) : (
                          <>
                            <Zap className="w-5 h-5 text-amber-300" />
                            <span>RUN AI INFERENCE</span>
                          </>
                        )}
                      </button>

                      {[
                        { id: 1, label: 'SST', desc: 'Sea Surface Temperature', min: 20, max: 35, step: 0.1, val: sst, setVal: setSst, unit: '°C' },
                        { id: 2, label: 'SSS', desc: 'Sea Surface Salinity', min: 30, max: 40, step: 0.1, val: sss, setVal: setSss, unit: 'PSU' },
                        { id: 3, label: 'SSH', desc: 'Sea Surface Height', min: -1.5, max: 1.5, step: 0.02, val: ssh, setVal: setSsh, unit: 'm' },
                      ].map(inp => (
                        <div key={inp.id} className="bg-surface-white border border-glass-border p-3 space-y-2 rounded-lg">
                          <div className="flex justify-between items-center">
                            <div className="flex flex-col">
                              <span className="font-headline-md text-[13px] text-on-surface">{inp.label}</span>
                            </div>
                            <div className="flex gap-1 items-center bg-surface-container-low px-2 py-1 border border-glass-border rounded">
                              <input 
                                type="number" 
                                value={(inp.val as any) === "" || (inp.val as any) === "-" ? (inp.val as any) : (isNaN(inp.val as any) ? "" : inp.val)} 
                                onChange={(e) => {
                                  const v = e.target.value;
                                  if (v === "" || v === "-") inp.setVal(v as any);
                                  else inp.setVal(parseFloat(v));
                                }}
                                onBlur={(e) => {
                                  let v = parseFloat(e.target.value);
                                  if (isNaN(v)) v = inp.min;
                                  inp.setVal(Math.min(Math.max(v, inp.min), inp.max));
                                }}
                                className="w-14 bg-transparent text-right font-label-mono text-on-surface outline-none"
                              />
                              <span className="font-label-mono text-[10px] text-text-muted">{inp.unit}</span>
                            </div>
                          </div>
                          <div className="relative pt-1">
                            <input 
                              type="range" min={inp.min} max={inp.max} step={inp.step} 
                              value={isNaN(inp.val) ? inp.min : inp.val} 
                              onChange={(e) => inp.setVal(parseFloat(e.target.value))}
                              onMouseUp={handleRunInference}
                              onTouchEnd={handleRunInference}
                              className="w-full h-1 bg-surface-container-high appearance-none outline-none accent-primary"
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })()
          )}"""

new_content = pattern.sub(replacement, content)
if new_content == content:
    print("Failed to replace!")
else:
    with open("frontend/src/app/page.tsx", "w") as f:
        f.write(new_content)
    print("Forecaster Map UI generated perfectly!")
