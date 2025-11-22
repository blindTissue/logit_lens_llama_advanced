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

  const [selectedModel, setSelectedModel] = useState("meta-llama/Llama-3.2-3B");
  const [lensType, setLensType] = useState("block_output");
  const [interventionType, setInterventionType] = useState<"scale" | "zero" | "block_attention">("zero");

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
    loadModel(newModel);
  };

  const runInference = async () => {
    try {
      setLoading(true);
      const res = await axios.post(`${API_URL}/inference`, {
        text,
        interventions,
        lens_type: lensType
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

  const addIntervention = (layer: string, type: "scale" | "zero" | "block_attention", value: number = 0, tokenIndex?: number, sourceTokens?: number[], targetTokens?: number[]) => {
    setInterventions(prev => ({
      ...prev,
      [layer]: { type, value, token_index: tokenIndex, source_tokens: sourceTokens, target_tokens: targetTokens }
    }));
  };

  const removeIntervention = (layer: string) => {
    setInterventions(prev => {
      const next = { ...prev };
      delete next[layer];
      return next;
    });
  };

  return (
    <div className="app-container">
      <header>
        <h1>LogitLens Llama Advanced</h1>
        <div className="header-controls flex-row" style={{ alignItems: 'center', gap: '1rem' }}>
          <select value={selectedModel} onChange={handleModelChange} disabled={loading}>
            <option value="meta-llama/Llama-3.2-3B">Llama 3.2 3B</option>
            <option value="meta-llama/Llama-3.2-1B">Llama 3.2 1B</option>
            <option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama 1.1B</option>
          </select>

          <select value={lensType} onChange={e => setLensType(e.target.value)} disabled={loading}>
            <option value="block_output">Block Output (Standard)</option>
            <option value="post_attention">Post-Attention</option>
            <option value="combined">Combined (Most Detailed)</option>
          </select>

          <div className="status">
            Status: {loading ? "Loading..." : modelLoaded ? "Ready" : "Not Loaded"}
          </div>
        </div>
      </header>

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
              <h3>Interventions</h3>
              {Object.entries(interventions).map(([layer, config]) => (
                <div key={layer} className="intervention-item flex-row" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
                  <span>
                    {layer}: {config.type}
                    {config.type === 'scale' && `(${config.value})`}
                    {config.type === 'block_attention' && config.source_tokens && config.target_tokens && ` (${config.source_tokens.join(',')} → ${config.target_tokens.join(',')})`}
                    {config.token_index !== undefined && ` @ Token ${config.token_index}`}
                  </span>
                  <button onClick={() => removeIntervention(layer)} style={{ padding: '4px 8px', fontSize: '0.8em' }}>X</button>
                </div>
              ))}

              <div className="add-intervention flex-row" style={{ marginTop: '1rem' }}>
                <select id="layer-select">
                  {/* Generate layer options dynamically based on loaded model */}
                  {results && results.logit_lens ? (
                    // Calculate number of layers from the logit lens results
                    Array.from({ length: Math.floor((results.logit_lens.length - 1) / (lensType === "combined" ? 2 : 1)) }, (_, i) => (
                      <option key={i} value={i}>Layer {i}</option>
                    ))
                  ) : (
                    // Default to 27 layers for Llama 3.2 3B if no results yet
                    Array.from({ length: 27 }, (_, i) => (
                      <option key={i} value={i}>Layer {i}</option>
                    ))
                  )}
                  <option value="embeddings">Embeddings</option>
                </select>

                <select id="location-select" style={{ display: interventionType === 'block_attention' ? 'none' : 'block' }}>
                  <option value="output">Block Output</option>
                  <option value="attn_output">Attn Output</option>
                  <option value="mlp_output">MLP Output</option>
                </select>

                <select id="type-select" value={interventionType} onChange={(e) => setInterventionType(e.target.value as "scale" | "zero" | "block_attention")}>
                  <option value="zero">Zero</option>
                  <option value="scale">Scale</option>
                  <option value="block_attention">Block Attention</option>
                </select>
                {interventionType !== 'block_attention' && (
                  <>
                    <input type="number" id="val-input" placeholder="Value" defaultValue={0} step={0.1} style={{ width: '60px' }} />
                    <input type="number" id="token-input" placeholder="Token Idx (opt)" style={{ width: '100px' }} />
                  </>
                )}
                {interventionType === 'block_attention' && (
                  <>
                    <input type="text" id="source-tokens-input" placeholder="Source tokens (e.g., 0,1)" style={{ width: '140px' }} />
                    <input type="text" id="target-tokens-input" placeholder="Target tokens (e.g., 3,4)" style={{ width: '140px' }} />
                  </>
                )}
                <button onClick={() => {
                  const layerSelect = document.getElementById('layer-select') as HTMLSelectElement;
                  const layer = layerSelect.value;

                  let interventionKey: string;
                  let sourceTokens: number[] | undefined;
                  let targetTokens: number[] | undefined;
                  let val = 0;
                  let tokenIndex: number | undefined;

                  if (interventionType === 'block_attention') {
                    // For attention blocking, use layer_X_attention format
                    if (layer === "embeddings") {
                      alert("Cannot block attention on embeddings");
                      return;
                    }
                    interventionKey = `layer_${layer}_attention`;

                    const sourceInput = (document.getElementById('source-tokens-input') as HTMLInputElement).value;
                    const targetInput = (document.getElementById('target-tokens-input') as HTMLInputElement).value;

                    if (!sourceInput || !targetInput) {
                      alert("Please specify both source and target tokens");
                      return;
                    }

                    sourceTokens = sourceInput.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
                    targetTokens = targetInput.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
                  } else {
                    // Regular intervention
                    const locationSelect = document.getElementById('location-select') as HTMLSelectElement;
                    val = parseFloat((document.getElementById('val-input') as HTMLInputElement).value);
                    const tokenInput = (document.getElementById('token-input') as HTMLInputElement).value;
                    tokenIndex = tokenInput ? parseInt(tokenInput) : undefined;

                    const location = locationSelect.value;
                    if (layer === "embeddings") {
                      interventionKey = "embeddings";
                    } else {
                      interventionKey = `layer_${layer}_${location}`;
                    }
                  }

                  addIntervention(interventionKey, interventionType, val, tokenIndex, sourceTokens, targetTokens);
                }}>Add</button>
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
                      {layer.predictions.map((p, i) => (
                        <td key={i} style={{ padding: '4px', borderBottom: '1px solid #333', textAlign: 'center' }}>
                          <div className="token-box" style={{
                            opacity: Math.max(0.3, p.prob * 2),
                            border: i === 0 ? '1px solid var(--primary-color)' : 'none',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            minWidth: '60px'
                          }}>
                            <span style={{ fontWeight: 'bold' }}>{p.token}</span>
                            <span style={{ fontSize: '0.7em', color: '#aaa' }}>{p.prob.toFixed(2)}</span>
                          </div>
                        </td>
                      ))}
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
    </div>
  )
}

export default App
