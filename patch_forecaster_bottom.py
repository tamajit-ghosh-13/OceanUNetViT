import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

# Locate the end of the `forecasting` tab layout
# It ends with:
#                     </div>
#                   </div>
#                 </div>
#               );
#             })()
#           )}

target = r"""                      ))}
                    </div>
                  </div>
                </div>
              );
            })()"""

if target not in content:
    print("Target not found!")
    exit(1)

# I will inject the Recharts graph and the Wargaming Simulator between the map/sidebar container and the closing div of the space-y-6 container.
new_content = r"""                      ))}
                    </div>
                  </div>

                  {/* BOTTOM SECTION: Analytics & Wargaming */}
                  <div className="flex flex-col lg:flex-row gap-6 mt-6">
                    {/* Live Recharts Graph */}
                    <div className="lg:w-1/2 flex flex-col border border-glass-border rounded-xl bg-background shadow-sm overflow-hidden">
                      <div className="bg-surface-container-low px-4 py-3 border-b border-glass-border flex justify-between items-center">
                        <span className="text-sm font-bold text-on-surface">Live Vertical Thermal Graph (Seed Coordinate)</span>
                        <span className="text-[10px] bg-rose-500 text-white px-2 py-0.5 rounded font-bold animate-pulse">LIVE RECHARTS</span>
                      </div>
                      <div className="p-4 h-[350px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart
                            layout="vertical"
                            data={inferResults?.depth_series?.filter((d:any) => d.depth_m <= 300) || []}
                            margin={{ top: 10, right: 10, left: -10, bottom: 0 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} stroke="#e5e7eb" />
                            <XAxis type="number" domain={[15, 35]} orientation="top" tick={{fontSize: 11}} />
                            <YAxis dataKey="depth_m" reversed={true} tick={{fontSize: 11}} domain={[0, 300]} type="number" />
                            <Tooltip 
                              cursor={{ stroke: '#f43f5e', strokeWidth: 1, strokeDasharray: '4 4' }}
                              contentStyle={{ backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: '8px', border: '1px solid #e5e7eb', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                              formatter={(value: any) => [`${Number(value).toFixed(2)}°C`, 'Temperature']} 
                              labelFormatter={(label: any) => `Depth: ${label}m`}
                            />
                            <ReferenceLine x={26} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 4" label={{ position: 'insideBottomRight', value: '26°C Fuel Threshold', fill: '#ef4444', fontSize: 11, fontWeight: 'bold' }} />
                            <Line type="monotone" dataKey="tribreed_degC" stroke="#f43f5e" strokeWidth={3} dot={{r: 0}} activeDot={{r: 6, strokeWidth: 0, fill: '#f43f5e'}} isAnimationActive={true} animationDuration={800} />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Wargaming Simulator */}
                    <div className="lg:w-1/2 bg-surface-white border border-glass-border p-7 flex flex-col shadow-sm rounded-xl justify-center">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="text-xl font-bold text-on-surface flex items-center gap-2.5">
                          <Crosshair className="w-5 h-5 text-rose-500" /> Interactive Wargaming Simulator
                        </h4>
                        <span className="bg-rose-500/10 text-rose-500 px-3 py-1 text-xs font-bold rounded-lg border border-rose-500/20">TRL-6 SCENARIO</span>
                      </div>
                      <p className="text-sm text-text-muted mb-6">
                        The MapGL physics engine predicts the trajectory, but what about Intensity? Based on the live UOHC ({tchp.toFixed(1)} kJ/cm²) at your selected coordinate, this simulator forecasts the explosive Rapid Intensification (RI) of your injected Seed Storm over 24 hours.
                      </p>
                      
                      <div className="bg-background border border-glass-border p-6 rounded-xl shadow-inner flex flex-col items-center justify-center">
                        {(() => {
                          let baseCat = seedStormCategory;
                          let futureCat = baseCat;
                          if (tchp < 20) futureCat = Math.max(0, baseCat - 1);
                          else if (tchp >= 50 && tchp < 80) futureCat = Math.min(5, baseCat + 1);
                          else if (tchp >= 80) futureCat = Math.min(5, baseCat + 2);
                          const catNames = ["Depression", "Category 1", "Category 2", "Category 3", "Category 4", "Category 5"];
                          
                          let description = "";
                          if (tchp < 20) description = "Insufficient deep thermal mass. Cyclone degrades.";
                          else if (tchp >= 20 && tchp < 50) description = "Stable heat availability. Storm maintains current intensity.";
                          else if (tchp >= 50 && tchp < 80) description = "High thermal fuel pool. Steady intensification expected.";
                          else if (tchp >= 80) description = "MASSIVE deep thermal fuel. Explosive Rapid Intensification triggered.";

                          return (
                            <div className="text-center w-full">
                              <div className="flex items-center justify-center gap-6 mb-4">
                                <div className="text-center">
                                  <div className="text-[10px] text-text-muted font-bold mb-1 uppercase">T=0h SEED</div>
                                  <div className="text-sm font-bold bg-surface-container px-4 py-2 rounded-lg border border-glass-border shadow-inner">{catNames[baseCat]}</div>
                                </div>
                                <ChevronRight className={`w-8 h-8 ${futureCat > baseCat ? "text-rose-500 animate-pulse" : "text-text-muted"}`} />
                                <div className="text-center">
                                  <div className="text-[10px] text-text-muted font-bold mb-1 uppercase">T+24h FORECAST</div>
                                  <div className={`text-sm font-bold px-4 py-2 rounded-lg border border-glass-border ${futureCat > baseCat ? "bg-rose-500 text-white shadow-lg shadow-rose-500/40" : "bg-surface-container"}`}>{catNames[futureCat]}</div>
                                </div>
                              </div>
                              <p className={`text-sm font-bold ${futureCat > baseCat ? "text-rose-500" : "text-text-muted"}`}>{description}</p>
                            </div>
                          );
                        })()}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()"""

content = content.replace(target, new_content)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)
print("Forecaster tab patched!")
