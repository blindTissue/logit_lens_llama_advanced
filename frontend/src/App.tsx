import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'
import type { InferenceResponse, Interventions } from './types'

const API_URL = 'http://localhost:8000';

function App() {
  const [text, setText] = useState("The capital of France is");
  const [interventions, setInterventions] = useState<Interventions>({});
  const [results, setResults] = useState<InferenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);

  const [selectedModel, setSelectedModel] = useState("meta-llama/Llama-3.2-1B");
  const [lensType, setLensType] = useState("block_output");
  const [attnAllLayers, setAttnAllLayers] = useState(false);
  const [useChatTemplate, setUseChatTemplate] = useState(false);


  const loadingRef = useRef(false);

  useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      const res = await axios.get(`${API_URL}/model_status`);
      if (res.data.loaded) {
        setModelLoaded(true);
        if (res.data.model_name) {
          setSelectedModel(res.data.model_name);
        }
      } else {
        loadModel(selectedModel);
      }
    } catch (err) {
      console.error("Failed to check status", err);
      loadModel(selectedModel);
    }
  };

  const loadModel = async (modelName: string) => {
    if (loadingRef.current) return;
    loadingRef.current = true;

    try {
      setLoading(true);
      await axios.post(`${API_URL}/load_model`, { model_name: modelName });
      setModelLoaded(true);
    } catch (err) {
      console.error("Failed to load model", err);
      alert("Failed to load model. Check backend console.");
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  };

  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value;
    setSelectedModel(newModel);
    setModelLoaded(false);

    // Auto-enable chat template for Instruct models
    if (newModel.toLowerCase().includes("instruct")) {
      setUseChatTemplate(true);
    } else {
      setUseChatTemplate(false);
    }

    loadModel(newModel);
  };

  const runInference = async () => {
    try {
      setLoading(true);
      const res = await axios.post(`${API_URL}/inference`, {
        text,
        interventions,
        lens_type: lensType,
        apply_chat_template: useChatTemplate
      });
      setResults(res.data);
    } catch (err: any) {
      console.error("Inference failed", err);
      if (err.response && err.response.status === 400 && err.response.data.detail === "Model not loaded") {
        alert("Model was not loaded (server might have restarted). Reloading model...");
        setModelLoaded(false);
        loadModel(selectedModel);
      } else {
        alert("Inference failed");
      }
    } finally {
      setLoading(false);
    }
  };

  const addIntervention = (layer: string, type: "scale" | "zero" | "block_attention", value: number = 0, tokenIndex?: number, sourceTokens?: number[], targetTokens?: number[], allLayers?: boolean) => {
    setInterventions(prev => ({
      ...prev,
      [layer]: { type, value, token_index: tokenIndex, source_tokens: sourceTokens, target_tokens: targetTokens, all_layers: allLayers }
    }));
  };

  const removeIntervention = (layer: string) => {
    setInterventions(prev => {
      const next = { ...prev };
      delete next[layer];
      return next;
    });
  };

  const [sessions, setSessions] = useState<{ id: string, name: string, timestamp: string }[]>([]);
  const [showLoadModal, setShowLoadModal] = useState(false);
  const [streamType, setStreamType] = useState<"zero" | "scale">("zero");

  const saveSession = async () => {
    const name = prompt("Enter a name for this session (optional):");
    if (name === null) return; // Cancelled

    try {
      await axios.post(`${API_URL}/save_state`, { name });
      alert("Session saved successfully!");
    } catch (err) {
      console.error("Failed to save session", err);
      alert("Failed to save session");
    }
  };

  const fetchSessions = async () => {
    try {
      const res = await axios.get(`${API_URL}/sessions`);
      setSessions(res.data);
    } catch (err) {
      console.error("Failed to fetch sessions", err);
    }
  };

  const openLoadModal = () => {
    fetchSessions();
    setShowLoadModal(true);
  };

  const loadSession = async (sessionId: string) => {
    try {
      const res = await axios.post(`${API_URL}/load_session`, { session_id: sessionId });
      const json = res.data;

      // Restore state
      setText(json.text);
      setInterventions(json.interventions);
      setLensType(json.lens_type);
      setResults(json.results);

      if (json.model_name && json.model_name !== selectedModel) {
        setSelectedModel(json.model_name);
        alert(`Session loaded. Model set to ${json.model_name}. Ensure this model is loaded if you want to run new inferences.`);
      }

      setShowLoadModal(false);
    } catch (err) {
      console.error("Failed to load session", err);
      alert("Failed to load session");
    }
  };

  const [tooltipData, setTooltipData] = useState<{
    layer: number;
    token: number;
    rect: DOMRect;
    preds: any[];
    layerName: string;
  } | null>(null);

  return (
    <div className="app-container">
      <header>
        <h1>LogitLens Llama Advanced</h1>
        <div className="header-controls flex-row" style={{ alignItems: 'center', gap: '1rem' }}>
          <select value={selectedModel} onChange={handleModelChange} disabled={loading}>
            <option value="meta-llama/Llama-3.2-3B">Llama 3.2 3B</option>
            <option value="meta-llama/Llama-3.2-1B">Llama 3.2 1B</option>
            <option value="meta-llama/Llama-3.2-1B-Instruct">Llama 3.2 1B Instruct</option>
            <option value="meta-llama/Llama-3.2-3B-Instruct">Llama 3.2 3B Instruct</option>
            <option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama 1.1B</option>
          </select>

          <label style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.9em' }}>
            <input
              type="checkbox"
              checked={useChatTemplate}
              onChange={(e) => setUseChatTemplate(e.target.checked)}
              disabled={loading}
            />
            Use Chat Template
          </label>

          <select value={lensType} onChange={e => setLensType(e.target.value)} disabled={loading}>
            <option value="block_output">Block Output (Standard)</option>
            <option value="post_attention">Post-Attention</option>
            <option value="combined">Combined (Most Detailed)</option>
          </select>

          <div className="status">
            Status: {loading ? "Loading..." : modelLoaded ? "Ready" : "Not Loaded"}
          </div>

          <div className="session-controls" style={{ display: 'flex', gap: '0.5rem' }}>
            <button onClick={saveSession} disabled={!results} className="secondary-btn" style={{ fontSize: '0.9em', padding: '4px 8px' }}>
              Save
            </button>
            <button onClick={openLoadModal} className="secondary-btn" style={{ fontSize: '0.9em', padding: '4px 8px' }}>
              Load
            </button>
          </div>
        </div>
      </header>

      {showLoadModal && (
        <div className="modal-overlay" style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.7)', zIndex: 1000,
          display: 'flex', justifyContent: 'center', alignItems: 'center'
        }}>
          <div className="modal-content" style={{
            backgroundColor: '#1a1a1a', padding: '20px', borderRadius: '8px',
            width: '400px', maxHeight: '80vh', overflowY: 'auto',
            border: '1px solid #333'
          }}>
            <h3 style={{ marginTop: 0 }}>Load Session</h3>
            <div className="session-list" style={{ display: 'flex', flexDirection: 'column', gap: '8px', margin: '15px 0' }}>
              {sessions.length === 0 ? (
                <p style={{ color: '#888' }}>No saved sessions found.</p>
              ) : (
                sessions.map(s => (
                  <button
                    key={s.id}
                    onClick={() => loadSession(s.id)}
                    className="secondary-btn"
                    style={{ textAlign: 'left', padding: '10px', display: 'flex', justifyContent: 'space-between' }}
                  >
                    <span>{s.name}</span>
                    <span style={{ fontSize: '0.8em', color: '#888' }}>{s.timestamp}</span>
                  </button>
                ))
              )}
            </div>
            <button onClick={() => setShowLoadModal(false)} style={{ width: '100%', padding: '8px' }}>
              Cancel
            </button>
          </div>
        </div>
      )}

      <main className="grid-cols-2">
        <div className="controls card">
          <h2>Controls</h2>
          <div className="flex-col">
            <label>
              Input Text:
              <textarea
                value={text}
                onChange={e => setText(e.target.value)}
                rows={4}
                style={{ width: '100%' }}
              />
            </label>

            <div className="interventions">
              <h3>Active Interventions</h3>
              {Object.entries(interventions).map(([layer, config]) => (
                <div key={layer} className="intervention-item flex-row" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
                  <span>
                    {layer}: {config.type}
                    {config.type === 'scale' && `(${config.value})`}
                    {config.type === 'block_attention' && config.source_tokens && config.target_tokens && ` (${config.source_tokens.join(',')} → ${config.target_tokens.join(',')})`}
                    {config.token_index != null && ` @ Token ${config.token_index}`}
                    {config.all_layers && ` [ALL LAYERS]`}
                  </span>
                  <button onClick={() => removeIntervention(layer)} style={{ padding: '4px 8px', fontSize: '0.8em' }}>X</button>
                </div>
              ))}

              {/* Stream Interventions Section */}
              <h4 style={{ marginTop: '1.5rem', marginBottom: '0.5rem' }}>Stream Interventions</h4>
              <p style={{ fontSize: '0.85em', color: '#888', marginBottom: '0.5rem' }}>
                Modify activation streams (zero out, scale, etc.)
              </p>
              <div className="add-intervention flex-row" style={{ marginTop: '0.5rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <select id="stream-layer-select">
                  {results && results.logit_lens ? (
                    Array.from({ length: Math.floor((results.logit_lens.length - 1) / (lensType === "combined" ? 2 : 1)) }, (_, i) => (
                      <option key={i} value={i}>Layer {i}</option>
                    ))
                  ) : (
                    Array.from({ length: 27 }, (_, i) => (
                      <option key={i} value={i}>Layer {i}</option>
                    ))
                  )}
                  <option value="embeddings">Embeddings</option>
                </select>

                <select id="stream-location-select">
                  <option value="output">Block Output</option>
                  <option value="attn_output">Attn Output</option>
                  <option value="mlp_output">MLP Output</option>
                </select>

                <select id="stream-type-select" value={streamType} onChange={(e) => setStreamType(e.target.value as "zero" | "scale")}>
                  <option value="zero">Zero</option>
                  <option value="scale">Scale</option>
                </select>

                {streamType === 'scale' && (
                  <input type="number" id="stream-val-input" placeholder="Value" defaultValue={0} step={0.1} style={{ width: '60px' }} />
                )}
                <input type="number" id="stream-token-input" placeholder="Token Idx (opt)" style={{ width: '100px' }} />
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <input type="checkbox" id="stream-all-layers" />
                  Apply to all layers
                </label>

                <button onClick={() => {
                  const layerSelect = document.getElementById('stream-layer-select') as HTMLSelectElement;
                  const locationSelect = document.getElementById('stream-location-select') as HTMLSelectElement;
                  // const typeSelect = document.getElementById('stream-type-select') as HTMLSelectElement; // No longer needed
                  const layer = layerSelect.value;
                  const location = locationSelect.value;
                  const type = streamType;

                  let val = 0;
                  if (type === 'scale') {
                    const valInput = document.getElementById('stream-val-input') as HTMLInputElement;
                    val = valInput ? parseFloat(valInput.value) : 0;
                  }
                  const tokenInput = (document.getElementById('stream-token-input') as HTMLInputElement).value;
                  const tokenIndex = tokenInput ? parseInt(tokenInput) : undefined;
                  const allLayers = (document.getElementById('stream-all-layers') as HTMLInputElement).checked;

                  let interventionKey: string;
                  if (allLayers) {
                    // Use generic key for all layers
                    interventionKey = `all_layers_${location}`;
                  } else if (layer === "embeddings") {
                    interventionKey = "embeddings";
                  } else {
                    interventionKey = `layer_${layer}_${location}`;
                  }

                  addIntervention(interventionKey, type, val, tokenIndex, undefined, undefined, allLayers);
                }} className="primary-btn">Add Stream Intervention</button>
              </div>

              {/* Attention Interventions Section */}
              <h4 style={{ marginTop: '1.5rem', marginBottom: '0.5rem' }}>Attention Interventions</h4>
              <p style={{ fontSize: '0.85em', color: '#888', marginBottom: '0.5rem' }}>
                Block attention connections between tokens
              </p>
              <div className="add-intervention flex-row" style={{ marginTop: '0.5rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <select id="attn-layer-select" disabled={attnAllLayers}>
                  {results && results.logit_lens ? (
                    Array.from({ length: Math.floor((results.logit_lens.length - 1) / (lensType === "combined" ? 2 : 1)) }, (_, i) => (
                      <option key={i} value={i}>Layer {i}</option>
                    ))
                  ) : (
                    Array.from({ length: 27 }, (_, i) => (
                      <option key={i} value={i}>Layer {i}</option>
                    ))
                  )}
                </select>

                <input type="text" id="attn-source-tokens-input" placeholder="Source tokens (e.g., 0)" style={{ width: '140px' }} />
                <input type="text" id="attn-target-tokens-input" placeholder="Target tokens (e.g., 3)" style={{ width: '140px' }} />
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <input
                    type="checkbox"
                    id="attn-all-layers"
                    checked={attnAllLayers}
                    onChange={(e) => setAttnAllLayers(e.target.checked)}
                  />
                  Apply to all layers
                </label>

                <button onClick={() => {
                  const layerSelect = document.getElementById('attn-layer-select') as HTMLSelectElement;
                  const layer = layerSelect.value;
                  const sourceInput = (document.getElementById('attn-source-tokens-input') as HTMLInputElement).value;
                  const targetInput = (document.getElementById('attn-target-tokens-input') as HTMLInputElement).value;

                  if (!sourceInput || !targetInput) {
                    alert("Please specify both source and target tokens");
                    return;
                  }

                  const sourceTokens = sourceInput.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
                  const targetTokens = targetInput.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
                  const allLayers = (document.getElementById('attn-all-layers') as HTMLInputElement).checked;

                  // Validate causal attention: source must be <= target (can't attend to future tokens)
                  for (const src of sourceTokens) {
                    for (const tgt of targetTokens) {
                      if (src > tgt) {
                        alert(`Warning: Source token ${src} > target token ${tgt}. In causal attention, tokens can only attend to current or previous positions (source <= target).`);
                        return;
                      }
                    }
                  }

                  // Use unique key for all layers to allow multiple interventions
                  // Format: all_layers_attention_0_1 (source_target)
                  let interventionKey: string;
                  if (allLayers) {
                    const srcStr = sourceTokens.join('_');
                    const tgtStr = targetTokens.join('_');
                    interventionKey = `all_layers_attention_${srcStr}_to_${tgtStr}`;
                  } else {
                    interventionKey = `layer_${layer}_attention`;
                  }
                  addIntervention(interventionKey, "block_attention", 0, undefined, sourceTokens, targetTokens, allLayers);
                }} className="primary-btn">Block Attention Flow</button>
              </div>
            </div>

            <button onClick={runInference} disabled={loading || !modelLoaded} className="primary-btn">
              Run LogitLens
            </button>
          </div>
        </div>

        <div className="visualization card">
          <h2>Visualization</h2>
          {results ? (
            <div className="logit-lens-view" style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', width: '100%' }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #333', minWidth: '120px', position: 'sticky', left: 0, background: 'var(--surface-color)', zIndex: 1 }}>Layer</th>
                    {results.logit_lens[0]?.predictions.map((_, i) => (
                      <th key={i} style={{ padding: '8px', borderBottom: '1px solid #333', minWidth: '80px', textAlign: 'center' }}>
                        Pos {i}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.logit_lens.map((layer) => (
                    <tr key={layer.layer_index} className="heatmap-row">
                      <td style={{ padding: '8px', borderBottom: '1px solid #333', fontSize: '0.9em', color: '#888', position: 'sticky', left: 0, background: 'var(--surface-color)', zIndex: 1 }}>
                        {layer.layer_name}
                      </td>
                      {layer.predictions.map((preds, i) => {
                        // preds is now an array of 5 predictions for this token position
                        const topPred = preds[0]; // Show top prediction in the cell
                        return (
                          <td
                            key={i}
                            style={{ padding: '4px', borderBottom: '1px solid #333', textAlign: 'center', position: 'relative' }}
                            onMouseEnter={(e) => {
                              const rect = e.currentTarget.getBoundingClientRect();
                              setTooltipData({
                                layer: layer.layer_index,
                                token: i,
                                rect,
                                preds,
                                layerName: layer.layer_name
                              });
                            }}
                            onMouseLeave={() => setTooltipData(null)}
                          >
                            <div className="token-box" style={{
                              opacity: Math.max(0.3, topPred.prob * 2),
                              border: i === 0 ? '1px solid var(--primary-color)' : 'none',
                              display: 'flex',
                              flexDirection: 'column',
                              alignItems: 'center',
                              minWidth: '60px'
                            }}>
                              <span style={{ fontWeight: 'bold' }}>{topPred.token}</span>
                              <span style={{ fontSize: '0.7em', color: '#aaa' }}>{topPred.prob.toFixed(2)}</span>
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="placeholder">Run inference to see results</div>
          )}
        </div>
      </main>

      {/* Global Fixed Tooltip */}
      {tooltipData && (() => {
        const TOOLTIP_WIDTH = 200; // approximate width
        const TOOLTIP_HEIGHT = 150; // approximate height
        const VIEWPORT_PADDING = 10;

        let top = tooltipData.rect.bottom + 4;
        let left = tooltipData.rect.left + (tooltipData.rect.width / 2);

        // Vertical boundary check
        if (top + TOOLTIP_HEIGHT > window.innerHeight) {
          top = tooltipData.rect.top - TOOLTIP_HEIGHT - 4; // Show above
        }

        // Horizontal boundary check
        if (left + (TOOLTIP_WIDTH / 2) > window.innerWidth - VIEWPORT_PADDING) {
          left = window.innerWidth - (TOOLTIP_WIDTH / 2) - VIEWPORT_PADDING;
        } else if (left - (TOOLTIP_WIDTH / 2) < VIEWPORT_PADDING) {
          left = (TOOLTIP_WIDTH / 2) + VIEWPORT_PADDING;
        }

        return (
          <div style={{
            position: 'fixed',
            top: top,
            left: left,
            transform: 'translateX(-50%)',
            backgroundColor: '#1a1a1a',
            border: '1px solid #444',
            borderRadius: '4px',
            padding: '8px',
            zIndex: 9999, // High z-index to ensure it's on top
            minWidth: '180px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
            pointerEvents: 'none', // Prevent tooltip from interfering with mouse events
            whiteSpace: 'nowrap'
          }}>
            <div style={{ fontSize: '0.8em', fontWeight: 'bold', marginBottom: '6px', color: '#aaa' }}>
              {tooltipData.layerName} - Pos {tooltipData.token}
            </div>
            {tooltipData.preds.map((pred, idx) => (
              <div key={idx} style={{
                fontSize: '0.75em',
                padding: '2px 0',
                display: 'flex',
                justifyContent: 'space-between',
                gap: '12px',
                color: idx === 0 ? '#fff' : '#ccc'
              }}>
                <span style={{ fontWeight: idx === 0 ? 'bold' : 'normal' }}>
                  {idx + 1}. {pred.token}
                </span>
                <span style={{ color: '#888' }}>{(pred.prob * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        );
      })()}
    </div>
  )
}

export default App
