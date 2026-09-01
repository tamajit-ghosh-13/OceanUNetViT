import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

target = r"""                          <div className="flex justify-between text-xs md:text-sm font-label-mono text-text-muted mt-2.5 px-1 font-medium">
                            <span>0 (Safe)</span>
                            <span>20 (Moderate)</span>
                            <span>50 (High / RI Threshold)</span>
                            <span>80 (Extreme Cat 4/5)</span>
                            <span>120+</span>
                          </div>
                        </div>
                      </div>"""

injection = r"""                          <div className="flex justify-between text-xs md:text-sm font-label-mono text-text-muted mt-2.5 px-1 font-medium">
                            <span>0 (Safe)</span>
                            <span>20 (Moderate)</span>
                            <span>50 (High / RI Threshold)</span>
                            <span>80 (Extreme Cat 4/5)</span>
                            <span>120+</span>
                          </div>
                        </div>
                      </div>

                      {/* MID SECTION: Cyclone Wargaming Simulator */}
                      <div className="bg-surface-white border border-glass-border p-7 flex flex-col shadow-md rounded-xl">
                        <div className="flex items-center justify-between mb-4">
                          <h4 className="text-xl font-bold text-on-surface flex items-center gap-2.5">
                            <Crosshair className="w-5 h-5 text-rose-500" /> Interactive Wargaming: Intensification Simulator
                          </h4>
                          <span className="bg-rose-500/10 text-rose-500 px-3 py-1 text-xs font-bold rounded-lg border border-rose-500/20">LIVE SCENARIO</span>
                        </div>
                        <p className="text-sm text-text-muted mb-6">
                          Select a theoretical &quot;Seed Storm&quot; entering this coordinate. The AI uses the live UOHC ({tchp.toFixed(1)} kJ/cm²) to forecast its explosive rapid intensification trajectory over 24 hours.
                        </p>
                        
                        <div className="flex flex-col md:flex-row gap-8 items-center bg-background border border-glass-border p-6 rounded-xl">
                          <div className="w-full md:w-1/2 flex flex-col gap-3">
                            <label className="text-sm font-semibold text-on-surface">Inject Seed Storm Category:</label>
                            <input 
                              type="range" 
                              min="0" max="3" step="1" 
                              value={seedStormCategory} 
                              onChange={(e) => setSeedStormCategory(parseInt(e.target.value))}
                              className="w-full accent-primary" 
                            />
                            <div className="flex justify-between text-xs text-text-muted font-bold">
                              <span>Depression</span>
                              <span>Cat 1</span>
                              <span>Cat 2</span>
                              <span>Cat 3</span>
                            </div>
                          </div>
                          
                          <div className="w-full md:w-1/2 flex items-center justify-center p-4 bg-surface-white border border-glass-border rounded-lg shadow-inner">
                            {(() => {
                              // Simplified Rapid Intensification Logic
                              let baseCat = seedStormCategory;
                              let futureCat = baseCat;
                              let description = "";
                              
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
                                  <div className="flex items-center justify-center gap-4 mb-2">
                                    <div className="text-center">
                                      <div className="text-xs text-text-muted mb-1 uppercase tracking-wider font-bold">T=0h</div>
                                      <div className="text-sm font-bold bg-surface-container px-3 py-1 rounded border border-glass-border text-on-surface">{startName}</div>
                                    </div>
                                    <ChevronRight className={`w-6 h-6 ${futureCat > baseCat ? "text-rose-500 animate-pulse" : "text-text-muted"}`} />
                                    <div className="text-center">
                                      <div className="text-xs text-text-muted mb-1 uppercase tracking-wider font-bold">T+24h</div>
                                      <div className={`text-sm font-bold px-3 py-1 rounded border border-glass-border ${futureCat > baseCat ? "bg-rose-500 text-white shadow-lg shadow-rose-500/20" : "bg-surface-container text-on-surface"}`}>{endName}</div>
                                    </div>
                                  </div>
                                  <p className={`text-xs mt-3 font-medium ${futureCat > baseCat ? "text-rose-500" : "text-text-muted"}`}>{description}</p>
                                </div>
                              );
                            })()}
                          </div>
                        </div>
                      </div>"""

content = content.replace(target, injection)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)
print("Cyclone UI patched successfully for Wargaming Simulator!")
