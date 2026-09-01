import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

# 1. Add the cycloneTrack logic before the return statement
# Search for `return (` inside the default export component
# Usually it's around `return (\n  <div className="flex h-screen bg-background">`
track_logic = r"""  let cycloneTrack: any = null;
  if (activeTab === "forecasting" && inferResults) {
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
    
    cycloneTrack = {
      type: "Feature",
      geometry: { type: "LineString", coordinates }
    };
  }

  return ("""
content = content.replace("  return (", track_logic, 1)

# 2. Modify tab visibility for the top section
content = content.replace(
    '{activeTab === "live_infer" && (', 
    '{(activeTab === "live_infer" || activeTab === "forecasting") && ('
)

# 3. Add GeoJSON to the Map component
map_marker = r"""                    {!isNaN(lat) && !isNaN(lon) && (
                      <Marker 
                        longitude={Math.min(Math.max(lon, 45), 105)} """

map_track = r"""                    {(activeTab === "forecasting" && cycloneTrack) && (
                      <Source id="cyclone-track" type="geojson" data={cycloneTrack}>
                        <Layer 
                          id="cyclone-track-line" 
                          type="line" 
                          paint={{
                            "line-color": "#ef4444",
                            "line-width": 5,
                            "line-opacity": 0.8
                          }} 
                        />
                      </Source>
                    )}
                    
                    {!isNaN(lat) && !isNaN(lon) && (
                      <Marker 
                        longitude={Math.min(Math.max(lon, 45), 105)} """
content = content.replace(map_marker, map_track)


# 4. Wrap the bottom section in live_infer condition
grid_start = r"""                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Left Column: 15-Depth Vertical Temperature Table */}"""

grid_wrapped = r"""                {activeTab === "live_infer" && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Left Column: 15-Depth Vertical Temperature Table */}"""
content = content.replace(grid_start, grid_wrapped)

# Find the end of the bottom section (which ends just before `</div>\n            </div>\n          )}`)
# Let's find the specific ending `</div>\n            </div>\n          )}` block for the first tab
grid_end = r"""                </div>
              </div>
            </div>
          )}

          {/* TAB 1: EXECUTIVE OVERVIEW */}"""

grid_end_wrapped = r"""                </div>
                )}
                
                {activeTab === "forecasting" && (
                  <div className="bg-surface-white border border-glass-border p-7 flex flex-col shadow-md rounded-xl mt-6">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-xl font-bold text-on-surface flex items-center gap-2.5">
                        <TrendingUp className="w-5 h-5 text-rose-500" /> AI-Driven Trajectory Engine: Kinetic Decay Model
                      </h4>
                      <span className="bg-rose-500/10 text-rose-500 px-3 py-1 text-xs font-bold rounded-lg border border-rose-500/20">LIVE MAP INJECTION</span>
                    </div>
                    <p className="text-sm text-text-muted mb-6">
                      The cyclone path drawn on the map above is calculated live. It uses your injected Seed Storm Category as initial inertia, and the AI's deep thermal fuel prediction (UOHC) at the origin coordinate as the battery life. <strong>If UOHC is high, the storm reaches the coast. If UOHC is low, it dies in the ocean.</strong> Change the SST/SSH inputs above to watch the downstream path manipulate!
                    </p>
                    
                    <div className="flex flex-col md:flex-row gap-8 items-center bg-background border border-glass-border p-6 rounded-xl">
                      <div className="w-full md:w-1/2 flex flex-col gap-3">
                        <label className="text-sm font-semibold text-on-surface">Inject Seed Storm Category:</label>
                        <input 
                          type="range" 
                          min="1" max="5" step="1" 
                          value={seedStormCategory} 
                          onChange={(e) => setSeedStormCategory(parseInt(e.target.value))}
                          className="w-full accent-primary" 
                        />
                        <div className="flex justify-between text-xs text-text-muted font-bold">
                          <span>Cat 1</span>
                          <span>Cat 3</span>
                          <span>Cat 5</span>
                        </div>
                      </div>
                      
                      <div className="w-full md:w-1/2 flex items-center justify-center p-4 bg-surface-white border border-glass-border rounded-lg shadow-inner">
                        <div className="text-center font-serif text-lg text-on-surface">
                           <i>I(t)</i> = <i>I<sub>0</sub></i> × exp( -<i>k</i> · <i>t</i> / <i>UOHC<sub>AI</sub></i> )
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 1: EXECUTIVE OVERVIEW */}"""

content = content.replace(grid_end, grid_end_wrapped)

# 5. Remove the OLD "forecasting" tab block
old_tab_start = r"""          {/* TAB 6: CYCLONE & EDDY FORECASTING */}
          {activeTab === "forecasting" && ("""
# Since this block ends at TAB 7, we can regex substitute it out
pattern = re.compile(r"          \{\/\* TAB 6: CYCLONE & EDDY FORECASTING \*\/\}.*?(?=          \{\/\* TAB 7: IN-SITU BENCHMARKS \*\/\})", re.DOTALL)
content = pattern.sub("", content)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)
print("Forecaster UI hijacked successfully!")
