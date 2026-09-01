import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

target = r"""                        <p className="text-sm text-text-muted mb-6">
                          Select a theoretical &quot;Seed Storm&quot; entering this coordinate. The AI uses the live UOHC ({tchp.toFixed(1)} kJ/cm²) to forecast its explosive rapid intensification trajectory over 24 hours.
                        </p>
                        
                        <div className="flex flex-col md:flex-row gap-8 items-center bg-background border border-glass-border p-6 rounded-xl">"""

injection = r"""                        <p className="text-sm text-text-muted mb-4">
                          Select a theoretical &quot;Seed Storm&quot; entering this coordinate. The AI uses the live UOHC ({tchp.toFixed(1)} kJ/cm²) to forecast its explosive rapid intensification trajectory over 24 hours.
                        </p>
                        
                        {/* Climate Anomaly Injector */}
                        <div className="bg-surface-container-low border border-glass-border p-5 rounded-xl mb-6 shadow-inner relative overflow-hidden">
                          <div className="absolute top-0 left-0 w-1 h-full bg-rose-500"></div>
                          <h5 className="text-sm font-bold text-on-surface mb-4 flex items-center gap-2">
                            <Sliders className="w-4 h-4 text-primary" /> Live Climate Anomaly Injector
                          </h5>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div>
                              <div className="flex justify-between text-xs font-bold mb-2">
                                <span className="text-text-muted uppercase tracking-wider">Sea Surface Temp (SST)</span>
                                <span className="text-rose-500 font-label-mono text-sm">{sst.toFixed(1)}°C</span>
                              </div>
                              <input 
                                type="range" 
                                min="24" max="34" step="0.1" 
                                value={sst} 
                                onChange={(e) => setSst(parseFloat(e.target.value))}
                                onMouseUp={handleRunInference}
                                onTouchEnd={handleRunInference}
                                className="w-full accent-rose-500" 
                              />
                            </div>
                            <div>
                              <div className="flex justify-between text-xs font-bold mb-2">
                                <span className="text-text-muted uppercase tracking-wider">Sea Surface Height (SSH)</span>
                                <span className="text-blue-500 font-label-mono text-sm">{ssh.toFixed(2)}m</span>
                              </div>
                              <input 
                                type="range" 
                                min="-0.5" max="0.5" step="0.01" 
                                value={ssh} 
                                onChange={(e) => setSsh(parseFloat(e.target.value))}
                                onMouseUp={handleRunInference}
                                onTouchEnd={handleRunInference}
                                className="w-full accent-blue-500" 
                              />
                            </div>
                          </div>
                          <p className="text-[11px] text-text-muted mt-4 flex items-center gap-1 font-medium">
                            <Activity className="w-3 h-3 text-rose-500" /> Releasing the slider instantly triggers the PyTorch AI backend to recalculate the 3D physics column.
                          </p>
                        </div>
                        
                        <div className="flex flex-col md:flex-row gap-8 items-center bg-background border border-glass-border p-6 rounded-xl">"""

content = content.replace(target, injection)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)
print("Climate Anomaly Injector added successfully!")
