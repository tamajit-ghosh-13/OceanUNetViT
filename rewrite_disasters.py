import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

# 1. We need to locate the `{activeTab === "disaster_risk" && (` block.
start_str = '          {/* TAB: DISASTER RISK INTELLIGENCE */}\n          {activeTab === "disaster_risk" && (\n            <div className="space-y-6">'

# 2. We need to locate the end of this block. It ends right before `          {/* TAB 1: EXECUTIVE OVERVIEW */}`
end_str = '          {/* TAB 1: EXECUTIVE OVERVIEW */}'

start_idx = content.find(start_str)
end_idx = content.find(end_str)

if start_idx == -1 or end_idx == -1:
    print("Could not find block boundaries")
    exit(1)

new_block = r"""          {/* TAB: DISASTER RISK INTELLIGENCE */}
          {activeTab === "disaster_risk" && (
            <div className="space-y-6">
              <div className="bg-surface-white border border-glass-border rounded-xl shadow-lg overflow-hidden">
                <div className="p-7 border-b border-glass-border flex justify-between items-center bg-surface-container-low">
                  <div>
                    <h3 className="text-2xl font-bold text-on-surface flex items-center gap-2">
                      <ShieldAlert className="w-6 h-6 text-rose-500" /> Disaster Risk Intelligence
                    </h3>
                    <p className="text-text-muted text-sm mt-1">Live 3D integrations of the AI thermal column predicting basin-wide catastrophes.</p>
                  </div>
                  <span className="bg-rose-500/10 text-rose-500 font-mono text-xs px-3 py-1 font-bold border border-rose-500/20 rounded">
                    LAT {lat.toFixed(2)}°N, LON {lon.toFixed(2)}°E
                  </span>
                </div>
                
                {/* HORIZONTAL SUB-TABS */}
                <div className="flex border-b border-glass-border bg-background overflow-x-auto">
                  <button 
                    onClick={() => setActiveDisasterTab("cyclone")}
                    className={`flex-1 py-4 px-6 font-bold text-sm transition-all border-b-2 flex items-center justify-center gap-2 ${activeDisasterTab === "cyclone" ? "border-rose-500 text-rose-500 bg-rose-500/5" : "border-transparent text-text-muted hover:text-on-surface hover:bg-surface-container"}`}
                  >
                    <Wind className="w-4 h-4" /> Cyclone Intensification
                  </button>
                  <button 
                    onClick={() => setActiveDisasterTab("heatwave")}
                    className={`flex-1 py-4 px-6 font-bold text-sm transition-all border-b-2 flex items-center justify-center gap-2 ${activeDisasterTab === "heatwave" ? "border-orange-500 text-orange-500 bg-orange-500/5" : "border-transparent text-text-muted hover:text-on-surface hover:bg-surface-container"}`}
                  >
                    <ThermometerSun className="w-4 h-4" /> Marine Heatwave
                  </button>
                  <button 
                    onClick={() => setActiveDisasterTab("drought")}
                    className={`flex-1 py-4 px-6 font-bold text-sm transition-all border-b-2 flex items-center justify-center gap-2 ${activeDisasterTab === "drought" ? "border-blue-500 text-blue-500 bg-blue-500/5" : "border-transparent text-text-muted hover:text-on-surface hover:bg-surface-container"}`}
                  >
                    <Droplets className="w-4 h-4" /> Drought/Flood (IOD)
                  </button>
                  <button 
                    onClick={() => setActiveDisasterTab("algae")}
                    className={`flex-1 py-4 px-6 font-bold text-sm transition-all border-b-2 flex items-center justify-center gap-2 ${activeDisasterTab === "algae" ? "border-emerald-500 text-emerald-500 bg-emerald-500/5" : "border-transparent text-text-muted hover:text-on-surface hover:bg-surface-container"}`}
                  >
                    <Biohazard className="w-4 h-4" /> Toxic Algal Bloom
                  </button>
                </div>

                <div className="p-7">
                  {/* CYCLONE TAB */}
                  {activeDisasterTab === "cyclone" && (
                    <div className="space-y-8 animate-in fade-in duration-300">
                      
                      {/* Top Row: Recharts & Info */}
                      <div className="flex flex-col lg:flex-row gap-8">
                        <div className="flex-1 space-y-4">
                          <h4 className="text-xl font-bold text-rose-500 flex items-center gap-2">
                            <Wind className="w-5 h-5" /> Cyclone Rapid Intensification (UOHC Payload)
                          </h4>
                          <p className="text-on-surface leading-relaxed text-sm">
                            The AI identifies massive columns of thermal energy capable of driving Category-5 rapid intensification. A cyclone acts as a massive heat engine, drawing power exclusively from seawater that is 26°C or warmer.
                          </p>
                          <div className="bg-surface-container border border-glass-border p-4 rounded-xl shadow-inner font-mono text-primary font-semibold text-center">
                            UOHC = c_p ρ ∫(T(z) - 26)dz [Limits: D26 to 0]
                          </div>
                          
                          <div className="flex items-center gap-4 p-4 border border-rose-500/30 bg-rose-500/5 rounded-xl">
                            <div className="w-16 h-16 bg-rose-500 rounded-full flex items-center justify-center text-white font-bold text-2xl shadow-lg shadow-rose-500/40">
                              {(inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50).toFixed(0)}
                            </div>
                            <div>
                              <div className="text-xs text-rose-500 font-bold uppercase tracking-wider">Live UOHC Fuel Payload</div>
                              <div className="text-2xl font-bold text-on-surface">kJ/cm²</div>
                            </div>
                          </div>
                          
                          <button 
                            onClick={() => setSelectedDisasterImage({
                              title: "Cyclone Rapid Intensification Simulation",
                              subtitle: "Vertical Temperature Profile & Cumulative Subsurface UOHC Heat Potential Integration",
                              src: "/simulations/sim_cyclone.png",
                              formula: "UOHC = c_p × ρ × ∫ (T(z) - 26°C) dz"
                            })}
                            className="w-full py-3 bg-surface-container-high hover:bg-surface-container-highest text-on-surface font-bold rounded-xl border border-glass-border flex items-center justify-center gap-2 transition-colors shadow-sm"
                          >
                            <Activity className="w-4 h-4" /> View Full Physical Simulation
                          </button>
                        </div>
                        
                        {/* Live Recharts Graph */}
                        <div className="lg:w-1/2 flex flex-col border border-glass-border rounded-xl bg-background shadow-inner overflow-hidden">
                          <div className="bg-surface-container-low px-4 py-2 border-b border-glass-border flex justify-between items-center">
                            <span className="text-xs font-bold text-text-muted uppercase tracking-wider">Live Vertical Thermal Integration</span>
                            <span className="text-xs bg-rose-500 text-white px-2 py-0.5 rounded font-bold animate-pulse">LIVE</span>
                          </div>
                          <div className="p-4 h-[300px]">
                            <ResponsiveContainer width="100%" height="100%">
                              <ComposedChart
                                layout="vertical"
                                data={inferResults?.depth_series?.filter((d:any) => d.depth_m <= 150) || []}
                                margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                              >
                                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} stroke="#e5e7eb" />
                                <XAxis type="number" domain={[20, 'auto']} orientation="top" tick={{fontSize: 10}} />
                                <YAxis type="number" dataKey="depth_m" reversed={true} tick={{fontSize: 10}} />
                                <ReferenceLine x={26} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 4" label={{ position: 'insideBottomRight', value: '26°C D26 Fuel Threshold', fill: '#ef4444', fontSize: 10, fontWeight: 'bold' }} />
                                <Line type="monotone" dataKey="tribreed_degC" stroke="#f43f5e" strokeWidth={3} dot={false} isAnimationActive={true} animationDuration={800} />
                              </ComposedChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      </div>
                      
                      {/* MID SECTION: Cyclone Wargaming Simulator */}
                      <div className="bg-surface-container border border-glass-border p-7 flex flex-col shadow-sm rounded-xl">
                        <div className="flex items-center justify-between mb-4">
                          <h4 className="text-xl font-bold text-on-surface flex items-center gap-2.5">
                            <Crosshair className="w-5 h-5 text-rose-500" /> Interactive Wargaming: Intensification Simulator
                          </h4>
                          <span className="bg-rose-500 text-white px-3 py-1 text-xs font-bold rounded-lg shadow-md">TRL-6 SCENARIO</span>
                        </div>
                        <p className="text-sm text-on-surface mb-6">
                          Select a theoretical "Seed Storm" entering this coordinate. The AI uses the live UOHC ({(inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50).toFixed(1)} kJ/cm²) to forecast its explosive rapid intensification trajectory over 24 hours.
                        </p>
                        
                        <div className="flex flex-col md:flex-row gap-8 items-center bg-background border border-glass-border p-6 rounded-xl shadow-inner">
                          <div className="w-full md:w-1/2 flex flex-col gap-3">
                            <label className="text-sm font-bold text-on-surface">Inject Seed Storm Category:</label>
                            <input 
                              type="range" 
                              min="0" max="3" step="1" 
                              value={seedStormCategory} 
                              onChange={(e) => setSeedStormCategory(parseInt(e.target.value))}
                              className="w-full accent-rose-500 h-2 bg-surface-container rounded-lg appearance-none cursor-pointer" 
                            />
                            <div className="flex justify-between text-xs text-text-muted font-bold mt-1">
                              <span>Depression</span>
                              <span>Cat 1</span>
                              <span>Cat 2</span>
                              <span>Cat 3</span>
                            </div>
                          </div>
                          
                          <div className="w-full md:w-1/2 flex items-center justify-center p-5 bg-surface-white border border-glass-border rounded-xl shadow-md">
                            {(() => {
                              // Simplified Rapid Intensification Logic
                              let baseCat = seedStormCategory;
                              let futureCat = baseCat;
                              let description = "";
                              const tchp = inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50;
                              
                              if (tchp < 20) {
                                futureCat = Math.max(0, baseCat - 1);
                                description = "Insufficient deep thermal mass. Cyclone churns up cold water and degrades.";
                              } else if (tchp >= 20 && tchp < 50) {
                                futureCat = baseCat;
                                description = "Stable heat availability. Storm maintains current intensity.";
                              } else if (tchp >= 50 && tchp < 80) {
                                futureCat = Math.min(5, baseCat + 1);
                                description = "High thermal fuel pool detected. Steady intensification expected.";
                              } else if (tchp >= 80) {
                                futureCat = Math.min(5, baseCat + 2);
                                description = "MASSIVE deep thermal fuel. Explosive Rapid Intensification (RI) triggered.";
                              }
                              
                              const catNames = ["Tropical Depression", "Category 1", "Category 2", "Category 3", "Category 4", "Cat 5 Super Cyclone"];
                              const startName = catNames[baseCat];
                              const endName = catNames[futureCat];
                              
                              return (
                                <div className="text-center w-full">
                                  <div className="flex items-center justify-center gap-4 mb-3">
                                    <div className="text-center">
                                      <div className="text-xs text-text-muted mb-1 uppercase tracking-wider font-bold">T=0h</div>
                                      <div className="text-sm font-bold bg-surface-container px-4 py-2 rounded-lg border border-glass-border text-on-surface shadow-inner">{startName}</div>
                                    </div>
                                    <ChevronRight className={`w-8 h-8 ${futureCat > baseCat ? "text-rose-500 animate-bounce" : "text-text-muted"}`} />
                                    <div className="text-center">
                                      <div className="text-xs text-text-muted mb-1 uppercase tracking-wider font-bold">T+24h</div>
                                      <div className={`text-sm font-bold px-4 py-2 rounded-lg border border-glass-border ${futureCat > baseCat ? "bg-rose-500 text-white shadow-lg shadow-rose-500/40" : "bg-surface-container text-on-surface shadow-inner"}`}>{endName}</div>
                                    </div>
                                  </div>
                                  <p className={`text-xs mt-3 font-bold ${futureCat > baseCat ? "text-rose-500" : "text-text-muted"}`}>{description}</p>
                                </div>
                              );
                            })()}
                          </div>
                        </div>
                      </div>
                      
                    </div>
                  )}

                  {/* HEATWAVE TAB */}
                  {activeDisasterTab === "heatwave" && (
                    <div className="flex flex-col md:flex-row gap-8 animate-in fade-in duration-300">
                      <div className="flex-1 space-y-4">
                        <h4 className="text-xl font-bold text-orange-500 flex items-center gap-2">
                          <ThermometerSun className="w-5 h-5" /> Benthic Marine Heatwave
                        </h4>
                        <p className="text-on-surface leading-relaxed text-sm">
                          Marine Heatwaves (MHWs) are prolonged periods of extreme ocean warming. While satellites detect surface heat, they are completely blind to benthic (deep-water) heatwaves which cause mass mortality in coral reefs.
                        </p>
                        <div className="bg-surface-container border border-glass-border p-4 rounded-xl shadow-inner font-mono text-primary font-semibold text-center">
                          ΔT_50 = T_predicted(50m) - T_baseline(50m)
                        </div>
                        
                        <div className="flex items-center gap-4 p-4 border border-orange-500/30 bg-orange-500/5 rounded-xl">
                          <div className="w-16 h-16 bg-orange-500 rounded-full flex items-center justify-center text-white font-bold text-2xl shadow-lg shadow-orange-500/40">
                            +2.4
                          </div>
                          <div>
                            <div className="text-xs text-orange-500 font-bold uppercase tracking-wider">50m Temperature Anomaly</div>
                            <div className="text-2xl font-bold text-on-surface">°C</div>
                          </div>
                        </div>
                        
                        <button 
                          onClick={() => setSelectedDisasterImage({
                            title: "Subsurface Marine Heatwave (Benthic)",
                            subtitle: "Prolonged periods of extreme ocean warming causing mass mortality in coral reefs, invisible from the surface.",
                            src: "/simulations/sim_heatwave.png",
                            formula: "ΔT_50 = T_predicted(50m) - T_baseline(50m)"
                          })}
                          className="w-full py-3 bg-surface-container-high hover:bg-surface-container-highest text-on-surface font-bold rounded-xl border border-glass-border flex items-center justify-center gap-2 transition-colors shadow-sm"
                        >
                          <Activity className="w-4 h-4" /> View Full Physical Simulation
                        </button>
                      </div>
                      
                      <div className="md:w-1/2 flex items-center justify-center bg-background border border-glass-border rounded-xl shadow-inner p-4 overflow-hidden">
                         <img src="/simulations/sim_heatwave.png" className="w-full max-w-[400px] rounded-lg shadow-md hover:scale-105 transition-transform duration-500 cursor-pointer" onClick={() => setSelectedDisasterImage({ title: "Marine Heatwave", subtitle: "", src: "/simulations/sim_heatwave.png", formula: "" })} />
                      </div>
                    </div>
                  )}

                  {/* DROUGHT TAB */}
                  {activeDisasterTab === "drought" && (
                    <div className="flex flex-col md:flex-row gap-8 animate-in fade-in duration-300">
                      <div className="flex-1 space-y-4">
                        <h4 className="text-xl font-bold text-blue-500 flex items-center gap-2">
                          <Droplets className="w-5 h-5" /> Drought & Flood Precursor (IOD)
                        </h4>
                        <p className="text-on-surface leading-relaxed text-sm">
                          The Indian Ocean Dipole (IOD) acts as a master switch for the Asian Monsoon. We track massive underwater Kelvin Waves by monitoring the D20 Thermocline. When it gets pushed deep, floods occur. When it rises, droughts occur.
                        </p>
                        <div className="bg-surface-container border border-glass-border p-4 rounded-xl shadow-inner font-mono text-primary font-semibold text-center">
                          D20 = Depth(z) where T(z) = 20°C
                        </div>
                        
                        <div className="flex items-center gap-4 p-4 border border-blue-500/30 bg-blue-500/5 rounded-xl">
                          <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-2xl shadow-lg shadow-blue-500/40">
                            {(inferResults?.derived_physical_products?.thermocline_d20_depth_m ?? 100).toFixed(0)}
                          </div>
                          <div>
                            <div className="text-xs text-blue-500 font-bold uppercase tracking-wider">D20 Thermocline Depth</div>
                            <div className="text-2xl font-bold text-on-surface">meters</div>
                          </div>
                        </div>
                        
                        <button 
                          onClick={() => setSelectedDisasterImage({
                            title: "Drought & Flood Precursor (IOD)",
                            subtitle: "Tracking massive underwater Kelvin Waves by monitoring the D20 Thermocline, the master switch for the Asian Monsoon.",
                            src: "/simulations/sim_drought.png",
                            formula: "D20 = Depth(z) where T(z) = 20°C"
                          })}
                          className="w-full py-3 bg-surface-container-high hover:bg-surface-container-highest text-on-surface font-bold rounded-xl border border-glass-border flex items-center justify-center gap-2 transition-colors shadow-sm"
                        >
                          <Activity className="w-4 h-4" /> View Full Physical Simulation
                        </button>
                      </div>
                      
                      <div className="md:w-1/2 flex items-center justify-center bg-background border border-glass-border rounded-xl shadow-inner p-4 overflow-hidden">
                         <img src="/simulations/sim_drought.png" className="w-full max-w-[400px] rounded-lg shadow-md hover:scale-105 transition-transform duration-500 cursor-pointer" onClick={() => setSelectedDisasterImage({ title: "Drought Precursor", subtitle: "", src: "/simulations/sim_drought.png", formula: "" })} />
                      </div>
                    </div>
                  )}

                  {/* ALGAE TAB */}
                  {activeDisasterTab === "algae" && (
                    <div className="flex flex-col md:flex-row gap-8 animate-in fade-in duration-300">
                      <div className="flex-1 space-y-4">
                        <h4 className="text-xl font-bold text-emerald-500 flex items-center gap-2">
                          <Biohazard className="w-5 h-5" /> Toxic Algal Bloom (Stratification)
                        </h4>
                        <p className="text-on-surface leading-relaxed text-sm">
                          Harmful Algal Blooms (HABs) and 'Dead Zones' occur when the ocean becomes highly stratified. A shallow layer of light, hot water sits directly on top of heavy, cold water, trapping agricultural runoff.
                        </p>
                        <div className="bg-surface-container border border-glass-border p-4 rounded-xl shadow-inner font-mono text-primary font-semibold text-center">
                          MLD = Depth(z) where (ρ(z) - ρ_surf) {">"} Δρ_thresh
                        </div>
                        
                        <div className="flex items-center gap-4 p-4 border border-emerald-500/30 bg-emerald-500/5 rounded-xl">
                          <div className="w-16 h-16 bg-emerald-500 rounded-full flex items-center justify-center text-white font-bold text-2xl shadow-lg shadow-emerald-500/40">
                            20.0
                          </div>
                          <div>
                            <div className="text-xs text-emerald-500 font-bold uppercase tracking-wider">Mixed Layer Depth (MLD)</div>
                            <div className="text-2xl font-bold text-on-surface">meters</div>
                          </div>
                        </div>
                        
                        <button 
                          onClick={() => setSelectedDisasterImage({
                            title: "Toxic Algal Bloom (Hypoxic Stratification)",
                            subtitle: "Harmful Algal Blooms (HABs) and 'Dead Zones' occur when the ocean becomes highly stratified.",
                            src: "/simulations/sim_algae.png",
                            formula: "MLD = Depth(z) where (ρ(z) - ρ_surf) > Δρ_thresh"
                          })}
                          className="w-full py-3 bg-surface-container-high hover:bg-surface-container-highest text-on-surface font-bold rounded-xl border border-glass-border flex items-center justify-center gap-2 transition-colors shadow-sm"
                        >
                          <Activity className="w-4 h-4" /> View Full Physical Simulation
                        </button>
                      </div>
                      
                      <div className="md:w-1/2 flex items-center justify-center bg-background border border-glass-border rounded-xl shadow-inner p-4 overflow-hidden">
                         <img src="/simulations/sim_algae.png" className="w-full max-w-[400px] rounded-lg shadow-md hover:scale-105 transition-transform duration-500 cursor-pointer" onClick={() => setSelectedDisasterImage({ title: "Toxic Algal Bloom", subtitle: "", src: "/simulations/sim_algae.png", formula: "" })} />
                      </div>
                    </div>
                  )}

                </div>
              </div>
            </div>
          )}

""" + end_str

content = content[:start_idx] + new_block + content[end_idx + len(end_str):]

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)

print("Disaster tabs rebuilt successfully!")
