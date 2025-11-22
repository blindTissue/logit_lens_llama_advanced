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

  const [selectedModel, setSelectedModel] = useState("TinyLlama/TinyLlama-1.1B-Chat-v1.0");

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
        interventions
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

  const addIntervention = (layer: string, type: "scale" | "zero", value: number = 0, tokenIndex?: number) => {
    setInterventions(prev => ({
      ...prev,
      [layer]: { type, value, token_index: tokenIndex }
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
        <div className="header-controls flex-row" style={{ alignItems: 'center' }}>
          <select value={selectedModel} onChange={handleModelChange} disabled={loading}>
            <option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama 1.1B</option>
            <option value="meta-llama/Llama-3.2-1B">Llama 3.2 1B</option>
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
                    {config.token_index !== undefined && ` @ Token ${config.token_index}`}
                  </span>
                  <button onClick={() => removeIntervention(layer)} style={{ padding: '4px 8px', fontSize: '0.8em' }}>X</button>
                </div>
              ))}

              <div className="add-intervention flex-row" style={{ marginTop: '1rem' }}>
                <select id="layer-select">
                  {Array.from({ length: 22 }, (_, i) => (
                    <option key={i} value={`layer_${i}_output`}>Layer {i} Output</option>
                  ))}
                  <option value="embeddings">Embeddings</option>
                </select>
                <select id="type-select">
                  <option value="zero">Zero</option>
                  <option value="scale">Scale</option>
                </select>
                <input type="number" id="val-input" placeholder="Value" defaultValue={0} step={0.1} style={{ width: '60px' }} />
                <input type="number" id="token-input" placeholder="Token Idx (opt)" style={{ width: '100px' }} />
                <button onClick={() => {
                  const layer = (document.getElementById('layer-select') as HTMLSelectElement).value;
                  const type = (document.getElementById('type-select') as HTMLSelectElement).value as "zero" | "scale";
                  const val = parseFloat((document.getElementById('val-input') as HTMLInputElement).value);
                  const tokenInput = (document.getElementById('token-input') as HTMLInputElement).value;
                  const tokenIndex = tokenInput ? parseInt(tokenInput) : undefined;
                  addIntervention(layer, type, val, tokenIndex);
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
