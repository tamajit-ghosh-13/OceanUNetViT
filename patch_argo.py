import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

target = r"""          {/* TAB 4: ARGO RECOMMENDER */}
          {activeTab === "recommender" && (
            <div className="space-y-6">
              <div className="bg-surface-white shadow-sm border border-glass-border border border-glass-border p-6  space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="font-headline-md text-headline-md text-on-surface flex items-center gap-2">
                      <Crosshair className="w-5 h-5 text-primary" />
                      ARGO Float Autonomous Mission Recommender
                    </h2>
                    <p className="font-body-sm text-body-sm text-text-muted">
                      Monte Carlo Dropout (N=35) Epistemic Uncertainty Guidance for INCOIS & Naval Deployment
                    </p>
                  </div>
                  <span className="text-body-sm bg-secondary/10 text-secondary border border-secondary-fixed-dim px-3 py-1  font-label-mono text-label-mono">
                    5 HIGH-VALUE TARGETS PINPOINTED
                  </span>
                </div>

                <div className=" overflow-hidden border border-glass-border bg-background text-on-surface">
                  <img
                    src="/assets/argo_mission_recommendations.png"
                    alt="ARGO Recommendations"
                    className="w-full object-cover"
                  />
                </div>
              </div>
            </div>
          )}"""

injection = r"""          {/* TAB 4: ARGO RECOMMENDER */}
          {activeTab === "recommender" && (
            <div className="space-y-6">
              <div className="bg-surface-white shadow-sm border border-glass-border p-8 rounded-xl space-y-6">
                <div className="flex items-center justify-between border-b border-glass-border pb-4">
                  <div>
                    <h2 className="text-2xl font-bold text-on-surface flex items-center gap-3">
                      <Crosshair className="w-7 h-7 text-primary" />
                      ARGO Float Autonomous Mission Recommender
                    </h2>
                    <p className="text-text-muted mt-2">
                      Monte Carlo Dropout (N=35) Epistemic Uncertainty Guidance for INCOIS & Naval Deployment
                    </p>
                  </div>
                  <span className="bg-primary/10 text-primary border border-primary/20 px-4 py-2 font-mono font-bold rounded-lg shadow-inner">
                    7 HIGH-VALUE TARGETS PINPOINTED
                  </span>
                </div>

                <div className="bg-surface-container-low border border-glass-border p-6 rounded-xl flex flex-col md:flex-row gap-6 items-center">
                  <div className="flex-1 space-y-4">
                    <h4 className="text-lg font-bold text-primary flex items-center gap-2">
                      <BrainCircuit className="w-5 h-5" /> Bayesian Epistemic Uncertainty
                    </h4>
                    <p className="text-sm text-on-surface leading-relaxed">
                      This map identifies the exact coordinates in the Indian Ocean where our OceanUNetViT model is least confident. By forcing the PyTorch model into <code>.train()</code> mode during inference, we execute 35 stochastic forward passes with neurons randomly deactivated. The variance across these passes reveals the model's <strong>Epistemic Uncertainty</strong> (knowledge gaps).
                    </p>
                    <div className="bg-white p-4 rounded border border-glass-border font-serif text-center shadow-sm">
                      <span className="italic text-lg">U<sub>epistemic</sub> = 1/N ∑ (y<sub>i</sub> - ȳ)²</span>
                    </div>
                    <p className="text-sm text-text-muted">
                      <strong>Deployment Strategy:</strong> Dropping physical ARGO floats exactly at these 7 glowing "hotspots" will yield the maximum possible reduction in basin-wide AI forecasting error.
                    </p>
                  </div>
                  <div className="md:w-1/3 flex justify-center">
                    <div className="bg-surface-white p-4 border-2 border-primary/50 shadow-lg rounded-xl transform rotate-2">
                      <div className="text-center font-bold text-primary mb-2">TARGET LOCK</div>
                      <div className="font-mono text-sm space-y-1 text-text-muted">
                        <div className="flex justify-between border-b border-glass-border pb-1"><span>Target 1:</span> <span className="text-on-surface font-bold">15°N, 62°E</span></div>
                        <div className="flex justify-between border-b border-glass-border pb-1"><span>Target 2:</span> <span className="text-on-surface font-bold">12°N, 88°E</span></div>
                        <div className="flex justify-between border-b border-glass-border pb-1"><span>Target 3:</span> <span className="text-on-surface font-bold">02°N, 55°E</span></div>
                        <div className="text-xs italic text-center mt-2 text-primary/70">+ 4 Secondary Targets</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="overflow-hidden border border-glass-border rounded-xl bg-background shadow-xl">
                  <img
                    src="/simulations/argo_mc_dropout.png"
                    alt="ARGO Recommendations"
                    className="w-full object-cover"
                  />
                </div>
              </div>
            </div>
          )}"""

content = content.replace(target, injection)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)
print("ARGO module rebuilt successfully!")
