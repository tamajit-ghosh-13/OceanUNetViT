import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

target = r"""                </>
              )}
            </div>
          )}

          {/* TAB 1: EXECUTIVE OVERVIEW */}"""

injection = r"""                </>
              )}

              {/* DISASTER RISK & ENVIRONMENTAL PHYSICS MODULE */}
              <div className="mt-8 bg-surface-white border border-glass-border p-7 rounded-xl shadow-lg">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-2xl font-bold text-on-surface flex items-center gap-2">
                      <ShieldAlert className="w-6 h-6 text-rose-500" /> Disaster Risk & Environmental Physics
                    </h3>
                    <p className="text-text-muted text-sm mt-1">Live 3D integrations of the AI thermal column predicting basin-wide catastrophes.</p>
                  </div>
                  <span className="bg-rose-500/10 text-rose-500 font-mono text-xs px-3 py-1 font-bold border border-rose-500/20 rounded">
                    LAT {lat.toFixed(2)}°N, LON {lon.toFixed(2)}°E
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  
                  {/* 1. Cyclone Intensification */}
                  <div 
                    className="border border-glass-border bg-background p-5 rounded-xl cursor-pointer hover:border-rose-500 transition-colors group"
                    onClick={() => setSelectedDisasterImage({
                      title: "Cyclone Rapid Intensification (UOHC Payload)",
                      subtitle: "The AI identifies massive columns of thermal energy capable of driving Category-5 rapid intensification.",
                      src: "/simulations/sim_cyclone.png",
                      formula: "UOHC = c_p ρ ∫(T(z) - 26)dz [Limits: D26 to 0]"
                    })}
                  >
                    <div className="flex justify-between items-start mb-3">
                      <h4 className="font-bold text-rose-500 flex items-center gap-2"><Wind className="w-4 h-4" /> Cyclone Rapid Intensification</h4>
                      <div className="w-8 h-8 rounded-full bg-rose-500/10 flex items-center justify-center text-rose-500 group-hover:bg-rose-500 group-hover:text-white transition-colors">↗</div>
                    </div>
                    <div className="flex items-center gap-4 mb-4">
                      <div className="w-16 h-16 bg-surface-container rounded-lg overflow-hidden border border-glass-border relative">
                        <img src="/simulations/sim_cyclone.png" className="w-full h-full object-cover opacity-80" />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                      </div>
                      <div className="flex-1">
                        <div className="text-xs text-text-muted font-bold mb-1 uppercase tracking-wider">UOHC Fuel Payload</div>
                        <div className="font-mono text-xl text-on-surface font-bold">{(inferResults.derived_physical_products?.tchp_kj_cm2 ?? 50).toFixed(1)} <span className="text-xs text-text-muted">kJ/cm²</span></div>
                      </div>
                    </div>
                    <div className="bg-surface-container-low p-3 rounded-lg border border-glass-border">
                      <p className="text-xs text-text-muted italic">"A cyclone acts as a massive heat engine, drawing power exclusively from seawater that is 26°C or warmer. To calculate the total available fuel, we integrate the excess heat..."</p>
                    </div>
                  </div>

                  {/* 2. Subsurface Marine Heatwave */}
                  <div 
                    className="border border-glass-border bg-background p-5 rounded-xl cursor-pointer hover:border-orange-500 transition-colors group"
                    onClick={() => setSelectedDisasterImage({
                      title: "Subsurface Marine Heatwave (Benthic)",
                      subtitle: "Prolonged periods of extreme ocean warming causing mass mortality in coral reefs, invisible from the surface.",
                      src: "/simulations/sim_heatwave.png",
                      formula: "ΔT_50 = T_predicted(50m) - T_baseline(50m)"
                    })}
                  >
                    <div className="flex justify-between items-start mb-3">
                      <h4 className="font-bold text-orange-500 flex items-center gap-2"><ThermometerSun className="w-4 h-4" /> Benthic Marine Heatwave</h4>
                      <div className="w-8 h-8 rounded-full bg-orange-500/10 flex items-center justify-center text-orange-500 group-hover:bg-orange-500 group-hover:text-white transition-colors">↗</div>
                    </div>
                    <div className="flex items-center gap-4 mb-4">
                      <div className="w-16 h-16 bg-surface-container rounded-lg overflow-hidden border border-glass-border relative">
                        <img src="/simulations/sim_heatwave.png" className="w-full h-full object-cover opacity-80" />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                      </div>
                      <div className="flex-1">
                        <div className="text-xs text-text-muted font-bold mb-1 uppercase tracking-wider">50m Temperature Anomaly</div>
                        <div className="font-mono text-xl text-on-surface font-bold">+2.4 <span className="text-xs text-text-muted">°C</span></div>
                      </div>
                    </div>
                    <div className="bg-surface-container-low p-3 rounded-lg border border-glass-border">
                      <p className="text-xs text-text-muted italic">"We define a benthic anomaly by subtracting a historical 10-year climatological baseline from the AI's real-time 3D prediction at the critical biological depth of 50 meters..."</p>
                    </div>
                  </div>

                  {/* 3. Drought/Flood (IOD) */}
                  <div 
                    className="border border-glass-border bg-background p-5 rounded-xl cursor-pointer hover:border-blue-500 transition-colors group"
                    onClick={() => setSelectedDisasterImage({
                      title: "Drought & Flood Precursor (IOD)",
                      subtitle: "Tracking massive underwater Kelvin Waves by monitoring the D20 Thermocline, the master switch for the Asian Monsoon.",
                      src: "/simulations/sim_drought.png",
                      formula: "D20 = Depth(z) where T(z) = 20°C"
                    })}
                  >
                    <div className="flex justify-between items-start mb-3">
                      <h4 className="font-bold text-blue-500 flex items-center gap-2"><Droplets className="w-4 h-4" /> Drought/Flood Precursor (IOD)</h4>
                      <div className="w-8 h-8 rounded-full bg-blue-500/10 flex items-center justify-center text-blue-500 group-hover:bg-blue-500 group-hover:text-white transition-colors">↗</div>
                    </div>
                    <div className="flex items-center gap-4 mb-4">
                      <div className="w-16 h-16 bg-surface-container rounded-lg overflow-hidden border border-glass-border relative">
                        <img src="/simulations/sim_drought.png" className="w-full h-full object-cover opacity-80" />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                      </div>
                      <div className="flex-1">
                        <div className="text-xs text-text-muted font-bold mb-1 uppercase tracking-wider">D20 Thermocline Depth</div>
                        <div className="font-mono text-xl text-on-surface font-bold">{(inferResults.derived_physical_products?.thermocline_d20_depth_m ?? 100).toFixed(0)} <span className="text-xs text-text-muted">meters</span></div>
                      </div>
                    </div>
                    <div className="bg-surface-container-low p-3 rounded-lg border border-glass-border">
                      <p className="text-xs text-text-muted italic">"When the D20 gets pushed deep, a massive pool of warm water builds up at the surface. This causes hyper-evaporation, fueling torrential rains and devastating floods..."</p>
                    </div>
                  </div>

                  {/* 4. Toxic Algal Bloom */}
                  <div 
                    className="border border-glass-border bg-background p-5 rounded-xl cursor-pointer hover:border-emerald-500 transition-colors group"
                    onClick={() => setSelectedDisasterImage({
                      title: "Toxic Algal Bloom (Hypoxic Stratification)",
                      subtitle: "Harmful Algal Blooms (HABs) and 'Dead Zones' occur when the ocean becomes highly stratified.",
                      src: "/simulations/sim_algae.png",
                      formula: "MLD = Depth(z) where (ρ(z) - ρ_surf) > Δρ_thresh"
                    })}
                  >
                    <div className="flex justify-between items-start mb-3">
                      <h4 className="font-bold text-emerald-500 flex items-center gap-2"><Biohazard className="w-4 h-4" /> Toxic Algal Bloom</h4>
                      <div className="w-8 h-8 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 group-hover:bg-emerald-500 group-hover:text-white transition-colors">↗</div>
                    </div>
                    <div className="flex items-center gap-4 mb-4">
                      <div className="w-16 h-16 bg-surface-container rounded-lg overflow-hidden border border-glass-border relative">
                        <img src="/simulations/sim_algae.png" className="w-full h-full object-cover opacity-80" />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                      </div>
                      <div className="flex-1">
                        <div className="text-xs text-text-muted font-bold mb-1 uppercase tracking-wider">Mixed Layer Depth (MLD)</div>
                        <div className="font-mono text-xl text-on-surface font-bold">20.0 <span className="text-xs text-text-muted">meters</span></div>
                      </div>
                    </div>
                    <div className="bg-surface-container-low p-3 rounded-lg border border-glass-border">
                      <p className="text-xs text-text-muted italic">"This sharp density jump acts like an impenetrable concrete ceiling. Agricultural fertilizers wash into the ocean and get trapped in this thin layer..."</p>
                    </div>
                  </div>

                </div>
              </div>
            </div>
          )}

          {/* TAB 1: EXECUTIVE OVERVIEW */}"""

content = content.replace(target, injection)

# Next, add the Modal at the very end of the file, just before </main>
modal_target = r"""</main>
      </div>
    </div>
  );
}"""

modal_injection = r"""      {/* DISASTER SIMULATION LIGHTBOX MODAL */}
      {selectedDisasterImage && (
        <div 
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 backdrop-blur-md p-4 md:p-8 animate-in fade-in duration-200" 
          onClick={() => setSelectedDisasterImage(null)}
        >
          <div 
            className="bg-surface-white border border-glass-border rounded-2xl shadow-2xl max-w-5xl w-full p-6 md:p-8 flex flex-col gap-4 relative"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center border-b border-glass-border pb-4">
              <div>
                <h3 className="text-xl md:text-2xl font-bold text-on-surface flex items-center gap-2.5">
                  <Activity className="w-6 h-6 text-primary" /> {selectedDisasterImage.title}
                </h3>
                <p className="text-sm md:text-base text-text-muted mt-1">{selectedDisasterImage.subtitle}</p>
              </div>
              <button 
                onClick={() => setSelectedDisasterImage(null)}
                className="text-text-muted hover:text-rose-400 p-2.5 rounded-xl bg-surface-container/50 hover:bg-surface-container transition-colors text-xl font-bold"
              >
                ✕
              </button>
            </div>

            <div className="bg-white rounded-xl p-4 flex items-center justify-center overflow-hidden border border-glass-border shadow-inner">
              <img 
                src={selectedDisasterImage.src} 
                alt={selectedDisasterImage.title} 
                className="w-full h-auto max-h-[75vh] object-contain rounded-lg shadow-sm"
              />
            </div>

            <div className="flex flex-wrap justify-between items-center text-xs md:text-sm text-text-muted pt-2 gap-2">
              <span className="font-mono text-primary font-semibold p-2 bg-primary/10 rounded">{selectedDisasterImage.formula}</span>
              <span>Click anywhere outside or ✕ to close</span>
            </div>
          </div>
        </div>
      )}
</main>
      </div>
    </div>
  );
}"""

content = content.replace(modal_target, modal_injection)

# Add missing Lucide icons to import
import_line = r"import { Anchor, Sparkles, MapPin, Zap, BrainCircuit, Activity, ChevronRight, Waves, ThermometerSun, Wind, Droplets, Biohazard, ShieldAlert"
if "ShieldAlert" not in content:
    content = content.replace(
        "import { Anchor, Sparkles, MapPin, Zap, BrainCircuit, Activity, ChevronRight",
        "import { Anchor, Sparkles, MapPin, Zap, BrainCircuit, Activity, ChevronRight, Wind, ThermometerSun, Droplets, Biohazard, ShieldAlert"
    )

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)
print("Disaster module injected successfully!")
