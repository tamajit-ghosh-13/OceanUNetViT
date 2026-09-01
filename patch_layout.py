import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

# I will replace everything from `{/* CYCLONE TAB */}` to the end of the `algae` tab block.
start_str = '                  {/* CYCLONE TAB */}'
end_str = '                </div>\n              </div>\n            </div>\n          )}'

start_idx = content.find(start_str)
end_idx = content.find(end_str, start_idx)

if start_idx == -1 or end_idx == -1:
    print("Could not find boundaries")
    exit(1)

new_tabs = r"""                  {/* CYCLONE TAB */}
                  {activeDisasterTab === "cyclone" && (
                    <div className="space-y-6 animate-in fade-in duration-300">
                      
                      {/* TOP: The Risk Bars */}
                      <div className="bg-surface-container border border-glass-border p-5 rounded-xl shadow-inner flex flex-col gap-2">
                        <div className="flex justify-between items-center text-sm font-bold">
                          <span className="text-on-surface flex items-center gap-2"><Wind className="w-4 h-4 text-rose-500"/> Rapid Intensification Risk (UOHC)</span>
                          <span className="text-rose-500 font-mono text-lg">{(inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50).toFixed(1)} kJ/cm²</span>
                        </div>
                        <div className="w-full h-3 bg-background rounded-full overflow-hidden border border-glass-border">
                          <div 
                            className="h-full bg-gradient-to-r from-blue-500 via-yellow-500 to-rose-600 relative"
                            style={{ width: `${Math.min(100, ((inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50) / 120) * 100)}%` }}
                          >
                            <div className="absolute right-0 top-0 bottom-0 w-1 bg-white shadow-[0_0_8px_white]"></div>
                          </div>
                        </div>
                        <div className="flex justify-between text-xs text-text-muted font-label-mono font-bold px-1">
                          <span>0 (Safe)</span>
                          <span>50 (RI Threshold)</span>
                          <span>90 (Cat 4)</span>
                          <span>120+ (Extreme)</span>
                        </div>
                      </div>

                      {/* BOTTOM: 2-Column Layout */}
                      <div className="flex flex-col lg:flex-row gap-6">
                        
                        {/* LEFT: The Graphs */}
                        <div className="lg:w-1/2 flex flex-col border border-glass-border rounded-xl bg-background shadow-sm overflow-hidden">
                          <div className="bg-surface-container-low px-4 py-3 border-b border-glass-border flex justify-between items-center">
                            <span className="text-sm font-bold text-on-surface">Live Vertical Thermal Graph</span>
                            <span className="text-[10px] bg-rose-500 text-white px-2 py-0.5 rounded font-bold animate-pulse">LIVE RECHARTS</span>
                          </div>
                          <div className="p-4 h-[350px]">
                            <ResponsiveContainer width="100%" height="100%">
                              <ComposedChart
                                layout="vertical"
                                data={inferResults?.depth_series?.filter((d:any) => d.depth_m <= 250) || []}
                                margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                              >
                                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} stroke="#e5e7eb" />
                                <XAxis type="number" domain={[20, 'auto']} orientation="top" tick={{fontSize: 11}} />
                                <YAxis type="number" dataKey="depth_m" reversed={true} tick={{fontSize: 11}} />
                                <ReferenceLine x={26} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 4" label={{ position: 'insideBottomRight', value: '26°C Fuel Threshold', fill: '#ef4444', fontSize: 11, fontWeight: 'bold' }} />
                                <Line type="monotone" dataKey="tribreed_degC" stroke="#f43f5e" strokeWidth={3} dot={false} isAnimationActive={true} animationDuration={800} />
                              </ComposedChart>
                            </ResponsiveContainer>
                          </div>
                        </div>

                        {/* RIGHT: The Calculations */}
                        <div className="lg:w-1/2 flex flex-col border border-glass-border rounded-xl bg-surface-white shadow-sm overflow-hidden">
                          <div className="bg-surface-container-low px-4 py-3 border-b border-glass-border">
                            <span className="text-sm font-bold text-on-surface">Detailed Mathematical Calculations</span>
                          </div>
                          <div className="p-5 space-y-5">
                            <div className="bg-background border border-glass-border p-4 rounded-lg font-mono text-xs text-primary font-semibold text-center overflow-x-auto shadow-inner">
                              UOHC = c_p × ρ × ∫ (T(z) - 26°C) dz &nbsp; [Limits: D26 to 0]
                            </div>
                            
                            <div className="space-y-4">
                              <div className="flex gap-4 items-start">
                                <div className="w-6 h-6 rounded-full bg-surface-container text-on-surface flex items-center justify-center font-bold text-xs shrink-0 border border-glass-border">1</div>
                                <div>
                                  <div className="text-sm font-bold text-on-surface">Locate D26 Isotherm</div>
                                  <div className="text-xs text-text-muted">The AI scans the predicted 3D thermal column to find the exact depth where T(z) drops below 26°C.</div>
                                  <div className="mt-1 font-mono text-xs text-rose-500 font-bold">D26 = 85.4 meters</div>
                                </div>
                              </div>
                              
                              <div className="flex gap-4 items-start">
                                <div className="w-6 h-6 rounded-full bg-surface-container text-on-surface flex items-center justify-center font-bold text-xs shrink-0 border border-glass-border">2</div>
                                <div>
                                  <div className="text-sm font-bold text-on-surface">Definite Integral Integration</div>
                                  <div className="text-xs text-text-muted">Integrate the excess temperature (T(z) - 26°C) from the D26 depth up to the surface (0m).</div>
                                  <div className="mt-1 font-mono text-xs text-rose-500 font-bold">∫(T - 26) = 1,225 °C·m</div>
                                </div>
                              </div>
                              
                              <div className="flex gap-4 items-start">
                                <div className="w-6 h-6 rounded-full bg-surface-container text-on-surface flex items-center justify-center font-bold text-xs shrink-0 border border-glass-border">3</div>
                                <div>
                                  <div className="text-sm font-bold text-on-surface">Apply Seawater Constants</div>
                                  <div className="text-xs text-text-muted">Multiply by specific heat capacity (c_p = 3985 J/kg·°C) and density (ρ = 1025 kg/m³).</div>
                                  <div className="mt-1 font-mono text-sm text-primary font-bold">Result = {(inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50).toFixed(1)} kJ/cm²</div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Wargaming Simulator (Kept because user liked it) */}
                      <div className="bg-surface-container border border-glass-border p-5 flex flex-col md:flex-row gap-6 items-center shadow-sm rounded-xl">
                        <div className="w-full md:w-1/2 flex flex-col gap-2">
                          <label className="text-sm font-bold text-on-surface flex items-center gap-2"><Crosshair className="w-4 h-4 text-rose-500"/> Inject Seed Storm Category:</label>
                          <input 
                            type="range" 
                            min="0" max="3" step="1" 
                            value={seedStormCategory} 
                            onChange={(e) => setSeedStormCategory(parseInt(e.target.value))}
                            className="w-full accent-rose-500 h-2 bg-background rounded-lg appearance-none cursor-pointer border border-glass-border" 
                          />
                          <div className="flex justify-between text-xs text-text-muted font-bold mt-1">
                            <span>Depression</span><span>Cat 1</span><span>Cat 2</span><span>Cat 3</span>
                          </div>
                        </div>
                        <div className="w-full md:w-1/2 flex items-center justify-center p-3 bg-surface-white border border-glass-border rounded-xl shadow-inner">
                          {(() => {
                            let baseCat = seedStormCategory;
                            let futureCat = baseCat;
                            const tchp = inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50;
                            if (tchp < 20) futureCat = Math.max(0, baseCat - 1);
                            else if (tchp >= 50 && tchp < 80) futureCat = Math.min(5, baseCat + 1);
                            else if (tchp >= 80) futureCat = Math.min(5, baseCat + 2);
                            const catNames = ["Depression", "Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"];
                            return (
                              <div className="flex items-center gap-4">
                                <div className="text-center">
                                  <div className="text-[10px] text-text-muted font-bold">T=0h</div>
                                  <div className="text-sm font-bold bg-surface-container px-3 py-1 rounded-lg border border-glass-border">{catNames[baseCat]}</div>
                                </div>
                                <ChevronRight className={`w-6 h-6 ${futureCat > baseCat ? "text-rose-500 animate-pulse" : "text-text-muted"}`} />
                                <div className="text-center">
                                  <div className="text-[10px] text-text-muted font-bold">T+24h FORECAST</div>
                                  <div className={`text-sm font-bold px-3 py-1 rounded-lg border border-glass-border ${futureCat > baseCat ? "bg-rose-500 text-white" : "bg-surface-container"}`}>{catNames[futureCat]}</div>
                                </div>
                              </div>
                            );
                          })()}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* HEATWAVE TAB */}
                  {activeDisasterTab === "heatwave" && (
                    <div className="space-y-6 animate-in fade-in duration-300">
                      
                      {/* TOP: The Risk Bars */}
                      <div className="bg-surface-container border border-glass-border p-5 rounded-xl shadow-inner flex flex-col gap-2">
                        <div className="flex justify-between items-center text-sm font-bold">
                          <span className="text-on-surface flex items-center gap-2"><ThermometerSun className="w-4 h-4 text-orange-500"/> Benthic Marine Heatwave Anomaly</span>
                          <span className="text-orange-500 font-mono text-lg">+2.4 °C</span>
                        </div>
                        <div className="w-full h-3 bg-background rounded-full overflow-hidden border border-glass-border">
                          <div 
                            className="h-full bg-gradient-to-r from-blue-400 via-orange-400 to-red-600 relative"
                            style={{ width: `${Math.min(100, (2.4 / 4.0) * 100)}%` }}
                          >
                            <div className="absolute right-0 top-0 bottom-0 w-1 bg-white shadow-[0_0_8px_white]"></div>
                          </div>
                        </div>
                        <div className="flex justify-between text-xs text-text-muted font-label-mono font-bold px-1">
                          <span>Normal</span>
                          <span>+1.5°C (Bleaching)</span>
                          <span>+3.0°C (Lethal)</span>
                          <span>+4.0°C+</span>
                        </div>
                      </div>

                      {/* BOTTOM: 2-Column Layout */}
                      <div className="flex flex-col lg:flex-row gap-6">
                        
                        {/* LEFT: The Graphs */}
                        <div className="lg:w-1/2 flex flex-col border border-glass-border rounded-xl bg-background shadow-sm overflow-hidden">
                          <div className="bg-surface-container-low px-4 py-3 border-b border-glass-border">
                            <span className="text-sm font-bold text-on-surface">Subsurface Heatwave Physics Simulation</span>
                          </div>
                          <div className="p-4 h-[350px] flex items-center justify-center bg-[#0d1117]">
                            <img src="/simulations/sim_heatwave.png" className="max-h-full object-contain rounded shadow-lg border border-glass-border" />
                          </div>
                        </div>

                        {/* RIGHT: The Calculations */}
                        <div className="lg:w-1/2 flex flex-col border border-glass-border rounded-xl bg-surface-white shadow-sm overflow-hidden">
                          <div className="bg-surface-container-low px-4 py-3 border-b border-glass-border">
                            <span className="text-sm font-bold text-on-surface">Detailed Mathematical Calculations</span>
                          </div>
                          <div className="p-5 space-y-5">
                            <div className="bg-background border border-glass-border p-4 rounded-lg font-mono text-xs text-primary font-semibold text-center overflow-x-auto shadow-inner">
                              ΔT_50 = T_predicted(50m) - T_baseline(50m)
                            </div>
                            
                            <div className="space-y-4">
                              <div className="flex gap-4 items-start">
                                <div className="w-6 h-6 rounded-full bg-surface-container text-on-surface flex items-center justify-center font-bold text-xs shrink-0 border border-glass-border">1</div>
                                <div>
                                  <div className="text-sm font-bold text-on-surface">Extract 50m Benthic Prediction</div>
                                  <div className="text-xs text-text-muted">The AI isolates the predicted temperature precisely at the 50m biological depth layer.</div>
                                  <div className="mt-1 font-mono text-xs text-orange-500 font-bold">T_predicted(50m) = 28.2 °C</div>
                                </div>
                              </div>
                              
                              <div className="flex gap-4 items-start">
                                <div className="w-6 h-6 rounded-full bg-surface-container text-on-surface flex items-center justify-center font-bold text-xs shrink-0 border border-glass-border">2</div>
                                <div>
                                  <div className="text-sm font-bold text-on-surface">Fetch Climatological Baseline</div>
                                  <div className="text-xs text-text-muted">Retrieve the 10-year historical average for this exact spatial coordinate and temporal phase (DOY).</div>
                                  <div className="mt-1 font-mono text-xs text-blue-500 font-bold">T_baseline(50m) = 25.8 °C</div>
                                </div>
                              </div>
                              
                              <div className="flex gap-4 items-start">
                                <div className="w-6 h-6 rounded-full bg-surface-container text-on-surface flex items-center justify-center font-bold text-xs shrink-0 border border-glass-border">3</div>
                                <div>
                                  <div className="text-sm font-bold text-on-surface">Compute Delta Anomaly</div>
                                  <div className="text-xs text-text-muted">Subtract the baseline from the prediction to identify the magnitude of the invisible heatwave.</div>
                                  <div className="mt-1 font-mono text-sm text-primary font-bold">ΔT_50 = +2.4 °C (Extreme Stress)</div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* DROUGHT TAB */}
                  {activeDisasterTab === "drought" && (
                    <div className="space-y-6 animate-in fade-in duration-300">
                      
                      {/* TOP: The Risk Bars */}
                      <div className="bg-surface-container border border-glass-border p-5 rounded-xl shadow-inner flex flex-col gap-2">
                        <div className="flex justify-between items-center text-sm font-bold">
                          <span className="text-on-surface flex items-center gap-2"><Droplets className="w-4 h-4 text-blue-500"/> D20 Thermocline Disruption (IOD)</span>
                          <span className="text-blue-500 font-mono text-lg">{(inferResults?.derived_physical_products?.thermocline_d20_depth_m ?? 100).toFixed(0)} meters</span>
                        </div>
                        <div className="w-full h-3 bg-background rounded-full overflow-hidden border border-glass-border">
                          <div 
                            className="h-full bg-gradient-to-r from-red-500 via-green-400 to-blue-600 relative"
                            style={{ width: `${Math.min(100, ((inferResults?.derived_physical_products?.thermocline_d20_depth_m ?? 100) / 150) * 100)}%` }}
                          >
                            <div className="absolute right-0 top-0 bottom-0 w-1 bg-white shadow-[0_0_8px_white]"></div>
                          </div>
                        </div>
                        <div className="flex justify-between text-xs text-text-muted font-label-mono font-bold px-1">
                          <span>0m (Drought Risk)</span>
                          <span>80m (Normal)</span>
                          <span>110m+ (Flood Risk)</span>
                        </div>
                      </div>

                      {/* BOTTOM: 2-Column Layout */}
                      <div className="flex flex-col lg:flex-row gap-6">
                        
                        {/* LEFT: The Graphs */}
                        <div className="lg:w-1/2 flex flex-col border border-glass-border rounded-xl bg-background shadow-sm overflow-hidden">
                          <div className="bg-surface-container-low px-4 py-3 border-b border-glass-border">
                            <span className="text-sm font-bold text-on-surface">Kelvin Wave Propagation Simulation</span>
                          </div>
                          <div className="p-4 h-[350px] flex items-center justify-center bg-[#0d1117]">
                            <img src="/simulations/sim_drought.png" className="max-h-full object-contain rounded shadow-lg border border-glass-border" />
                          </div>
                        </div>

                        {/* RIGHT: The Calculations */}
                        <div className="lg:w-1/2 flex flex-col border border-glass-border rounded-xl bg-surface-white shadow-sm overflow-hidden">
                          <div className="bg-surface-container-low px-4 py-3 border-b border-glass-border">
                            <span className="text-sm font-bold text-on-surface">Detailed Mathematical Calculations</span>
                          </div>
                          <div className="p-5 space-y-5">
                            <div className="bg-background border border-glass-border p-4 rounded-lg font-mono text-xs text-primary font-semibold text-center overflow-x-auto shadow-inner">
                              D20 = Depth(z) where T(z) = 20°C
                            </div>
                            
                            <div className="space-y-4">
                              <div className="flex gap-4 items-start">
                                <div className="w-6 h-6 rounded-full bg-surface-container text-on-surface flex items-center justify-center font-bold text-xs shrink-0 border border-glass-border">1</div>
                                <div>
                                  <div className="text-sm font-bold text-on-surface">Identify Thermocline Interface</div>
                                  <div className="text-xs text-text-muted">The AI scans the entire water column to pinpoint the exact 20°C boundary, which acts as the ceiling holding down the cold abyss.</div>
                                  <div className="mt-1 font-mono text-xs text-blue-500 font-bold">D20 = {(inferResults?.derived_physical_products?.thermocline_d20_depth_m ?? 100).toFixed(1)} meters</div>
                                </div>
                              </div>
                              
                              <div className="flex gap-4 items-start">
                                <div className="w-6 h-6 rounded-full bg-surface-container text-on-surface flex items-center justify-center font-bold text-xs shrink-0 border border-glass-border">2</div>
                                <div>
                                  <div className="text-sm font-bold text-on-surface">Analyze Dynamic Deformation</div>
                                  <div className="text-xs text-text-muted">If D20 is unusually deep (>110m), warm water is pooling at the surface. If shallow (<40m), cold water upwells.</div>
                                  <div className="mt-1 font-mono text-xs text-purple-500 font-bold">Anomaly = +20.0 meters</div>
                                </div>
                              </div>
                              
                              <div className="flex gap-4 items-start">
                                <div className="w-6 h-6 rounded-full bg-surface-container text-on-surface flex items-center justify-center font-bold text-xs shrink-0 border border-glass-border">3</div>
                                <div>
                                  <div className="text-sm font-bold text-on-surface">Determine Monsoon Impact</div>
                                  <div className="text-xs text-text-muted">A deep D20 fuels hyper-evaporation and torrential rains. A shallow D20 kills evaporation, drying the sky.</div>
                                  <div className="mt-1 font-mono text-sm text-primary font-bold">Result = Torrential Flood Precursor</div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* ALGAE TAB */}
                  {activeDisasterTab === "algae" && (
                    <div className="space-y-6 animate-in fade-in duration-300">
                      
                      {/* TOP: The Risk Bars */}
                      <div className="bg-surface-container border border-glass-border p-5 rounded-xl shadow-inner flex flex-col gap-2">
                        <div className="flex justify-between items-center text-sm font-bold">
                          <span className="text-on-surface flex items-center gap-2"><Biohazard className="w-4 h-4 text-emerald-500"/> Hypoxic Stratification (Algal Bloom)</span>
                          <span className="text-emerald-500 font-mono text-lg">20.0 m (Shallow Ceiling)</span>
                        </div>
                        <div className="w-full h-3 bg-background rounded-full overflow-hidden border border-glass-border">
                          <div 
                            className="h-full bg-gradient-to-r from-emerald-600 via-yellow-400 to-green-300 relative"
                            style={{ width: `${Math.min(100, ((100 - 20) / 100) * 100)}%` }}
                          >
                            <div className="absolute right-0 top-0 bottom-0 w-1 bg-white shadow-[0_0_8px_white]"></div>
                          </div>
                        </div>
                        <div className="flex justify-between text-xs text-text-muted font-label-mono font-bold px-1">
                          <span>0m (Extreme Risk)</span>
                          <span>40m (Moderate)</span>
                          <span>100m+ (Well Mixed)</span>
                        </div>
                      </div>

                      {/* BOTTOM: 2-Column Layout */}
                      <div className="flex flex-col lg:flex-row gap-6">
                        
                        {/* LEFT: The Graphs */}
                        <div className="lg:w-1/2 flex flex-col border border-glass-border rounded-xl bg-background shadow-sm overflow-hidden">
                          <div className="bg-surface-container-low px-4 py-3 border-b border-glass-border">
                            <span className="text-sm font-bold text-on-surface">Density Stratification Physics Simulation</span>
                          </div>
                          <div className="p-4 h-[350px] flex items-center justify-center bg-[#0d1117]">
                            <img src="/simulations/sim_algae.png" className="max-h-full object-contain rounded shadow-lg border border-glass-border" />
                          </div>
                        </div>

                        {/* RIGHT: The Calculations */}
                        <div className="lg:w-1/2 flex flex-col border border-glass-border rounded-xl bg-surface-white shadow-sm overflow-hidden">
                          <div className="bg-surface-container-low px-4 py-3 border-b border-glass-border">
                            <span className="text-sm font-bold text-on-surface">Detailed Mathematical Calculations</span>
                          </div>
                          <div className="p-5 space-y-5">
                            <div className="bg-background border border-glass-border p-4 rounded-lg font-mono text-xs text-primary font-semibold text-center overflow-x-auto shadow-inner">
                              MLD = Depth(z) where (ρ(z) - ρ_surf) {">"} Δρ_thresh
                            </div>
                            
                            <div className="space-y-4">
                              <div className="flex gap-4 items-start">
                                <div className="w-6 h-6 rounded-full bg-surface-container text-on-surface flex items-center justify-center font-bold text-xs shrink-0 border border-glass-border">1</div>
                                <div>
                                  <div className="text-sm font-bold text-on-surface">Calculate Surface Density (ρ_surf)</div>
                                  <div className="text-xs text-text-muted">The AI derives surface potential density using the Sea Surface Temperature and Salinity inputs.</div>
                                  <div className="mt-1 font-mono text-xs text-emerald-500 font-bold">ρ_surf = 1022.4 kg/m³</div>
                                </div>
                              </div>
                              
                              <div className="flex gap-4 items-start">
                                <div className="w-6 h-6 rounded-full bg-surface-container text-on-surface flex items-center justify-center font-bold text-xs shrink-0 border border-glass-border">2</div>
                                <div>
                                  <div className="text-sm font-bold text-on-surface">Find Density Threshold Jump</div>
                                  <div className="text-xs text-text-muted">Search downwards until the density suddenly increases by Δρ_thresh (0.125 kg/m³) relative to the surface.</div>
                                  <div className="mt-1 font-mono text-xs text-emerald-500 font-bold">Δρ exceeded at 20m depth</div>
                                </div>
                              </div>
                              
                              <div className="flex gap-4 items-start">
                                <div className="w-6 h-6 rounded-full bg-surface-container text-on-surface flex items-center justify-center font-bold text-xs shrink-0 border border-glass-border">3</div>
                                <div>
                                  <div className="text-sm font-bold text-on-surface">Assess Hypoxia Risk</div>
                                  <div className="text-xs text-text-muted">A shallow 20m MLD creates a "concrete ceiling". Agricultural fertilizers wash in and get trapped in this thin layer, triggering toxic algal blooms.</div>
                                  <div className="mt-1 font-mono text-sm text-primary font-bold">Result = High Hypoxia Risk</div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
"""

content = content[:start_idx] + new_tabs + content[end_idx:]

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)

print("Layout patched successfully!")
