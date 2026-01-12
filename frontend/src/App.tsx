import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'
import type { InferenceResponse, Interventions } from './types'
import { SearchableModelDropdown } from './SearchableModelDropdown'

// Simple Heatmap Component
const AttentionHeatmap = ({ data, tokens, title, size = 30, saveName }: { data: number[][], tokens: string[], title?: string, size?: number, saveName?: string }) => {
  if (!data || !data.length) return null;

  const handleSave = async () => {
    try {
      const res = await axios.post(`${API_URL}/save_visualization`, {
        attention_data: data,
        tokens: tokens,
        title: saveName || title || "Attention Heatmap"
      });
      alert(`Visualization saved to: ${res.data.filepath}`);
    } catch (err) {
      console.error("Failed to save visualization", err);
      alert("Failed to save visualization");
    }
  };

  return (
    <div className="attention-heatmap" style={{ overflowX: 'auto', marginTop: '10px', display: 'inline-block', marginRight: '10px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
        {title && <h4 style={{ margin: 0, fontSize: '0.9em' }}>{title}</h4>}
        <button onClick={handleSave} className="save-btn">Save</button>
      </div>
      <table style={{ borderCollapse: 'collapse', fontSize: '0.8em' }}>
        <thead>
          <tr>
            <th style={{ padding: '2px' }}></th>
            {tokens.map((tok, i) => (
              <th key={i} style={{ padding: '2px', writingMode: 'vertical-rl', transform: 'rotate(180deg)', minHeight: '40px', fontSize: '0.7em' }}>
                {tok.slice(0, 10)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              <td style={{ padding: '2px', textAlign: 'right', whiteSpace: 'nowrap', fontSize: '0.7em' }}>{tokens[i].slice(0, 10)}</td>
              {row.map((val, j) => (
                <td key={j}
                  title={`Src: ${tokens[j]}\nTgt: ${tokens[i]}\nAttn: ${val.toFixed(4)}`}
                  style={{
                    width: `${size}px`,
                    height: `${size}px`,
                    backgroundColor: `rgba(0, 255, 128, ${val})`, // Green heatmap
                    border: '1px solid #333',
                    fontSize: '0.6em',
                    color: val > 0.5 ? '#000' : '#fff',
                    textAlign: 'center',
                    verticalAlign: 'middle',
                    padding: '2px',
                  }}>
                  {size >= 25 && val.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const API_URL = 'http://localhost:8000';

/**
 * Check if a model is an instruct/chat model that should use chat templates.
 * - Llama: models with "Instruct" or "Chat" in name
 * - Qwen3: models WITHOUT "-Base" suffix (Qwen3 default models are instruct)
 */
function isInstructModel(modelName: string): boolean {
  // Explicit instruct/chat indicators
  if (modelName.includes("Instruct") || modelName.includes("Chat")) {
    return true;
  }
  // Qwen3 models: default (no suffix) are instruct, "-Base" are base models
  if (modelName.includes("Qwen3") && !modelName.includes("-Base")) {
    return true;
  }
  return false;
}

/**
 * Format a token for display, making whitespace characters visible.
 */
function formatToken(token: string): string {
  if (token === '\n') return '\\n';
  if (token === '\t') return '\\t';
  if (token === '\r') return '\\r';
  if (token === '') return '␣';
  if (token.trim() === '') return '␣';
  return token;
}

function App() {
  const [text, setText] = useState("The capital of France is");
  const [interventions, setInterventions] = useState<Interventions>({});
  const [results, setResults] = useState<InferenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState<string>("");
  const [modelLoaded, setModelLoaded] = useState(false);

  const [selectedModel, setSelectedModel] = useState("meta-llama/Llama-3.2-1B");
  const [selectedBackend, setSelectedBackend] = useState("custom");
  const [lensType, setLensType] = useState("block_output");
  const [attnAllLayers, setAttnAllLayers] = useState(false);
  const [streamAllLayers, setStreamAllLayers] = useState(false);
  const [useChatTemplate, setUseChatTemplate] = useState(false);
  const [showMoreModels, setShowMoreModels] = useState(false);

  // Attention State
  const [showAttention, setShowAttention] = useState(false);
  const [attnAggregation, setAttnAggregation] = useState("all_average"); // Default to "Average All"
  const [attnLayer, setAttnLayer] = useState(0);
  const [attnHead, setAttnHead] = useState(0);


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
        if (res.data.backend) {
          setSelectedBackend(res.data.backend);
        }
      } else {
        loadModel(selectedModel, selectedBackend);
      }
    } catch (err) {
      console.error("Failed to check status", err);
      loadModel(selectedModel, selectedBackend);
    }
  };

  const loadModel = async (modelName: string, backend: string = "custom"): Promise<boolean> => {
    if (loadingRef.current) return false;
    loadingRef.current = true;

    try {
      setLoading(true);
      setLoadingStatus(`Loading ${modelName}...`);
      await axios.post(`${API_URL}/load_model`, { model_name: modelName, backend: backend });
      setModelLoaded(true);
      setSelectedBackend(backend);
      setLoadingStatus("");
      return true;
    } catch (err: any) {
      console.error("Failed to load model", err);
      const errorMsg = err.response?.data?.detail || err.message || "Unknown error";
      alert(`Failed to load model "${modelName}".\n\nError: ${errorMsg}\n\nTips:\n- Check that the model name is correct (e.g., "EleutherAI/pythia-160m")\n- For TransformerLens backend, see supported models: https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html\n- Check backend console for detailed error logs`);
      setModelLoaded(false);
      setLoadingStatus("");
      return false;
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  };

  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value;

    // Check if user selected "more models" option
    if (newModel === "__more_models__") {
      setShowMoreModels(true);
      return; // Show the searchable dropdown instead
    }

    setShowMoreModels(false);
    setSelectedModel(newModel);
    setModelLoaded(false);

    // Auto-enable chat template for instruct/chat models, disable for base models
    setUseChatTemplate(isInstructModel(newModel));

    loadModel(newModel, selectedBackend);
  };

  const handleMoreModelsSelect = (modelId: string) => {
    setSelectedModel(modelId);
    setModelLoaded(false);
    setShowMoreModels(false);

    // Auto-enable chat template for instruct/chat models, disable for base models
    setUseChatTemplate(isInstructModel(modelId));

    loadModel(modelId, selectedBackend);
  };

  const handleBackendChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newBackend = e.target.value;
    setSelectedBackend(newBackend);
    setModelLoaded(false);
    setShowMoreModels(false); // Reset the searchable dropdown when changing backends

    // Check if current model is compatible with new backend
    let modelToLoad = selectedModel;

    // If switching to custom backend and currently on a TL-only model, switch to Llama
    if (newBackend === "custom" && !selectedModel.includes("Llama") && !selectedModel.includes("Qwen3")) {
      modelToLoad = "meta-llama/Llama-3.2-1B";
      setSelectedModel(modelToLoad);
    }

    // If switching to TL backend and currently on Qwen3, switch to Llama
    if (newBackend === "transformerlens" && selectedModel.includes("Qwen3")) {
      modelToLoad = "meta-llama/Llama-3.2-1B";
      setSelectedModel(modelToLoad);
    }

    loadModel(modelToLoad, newBackend);
  };

  const runInference = async (overrides?: {
    text?: string;
    interventions?: Interventions;
    lensType?: string;
    useChatTemplate?: boolean;
  }) => {
    try {
      setLoading(true);
      const res = await axios.post(`${API_URL}/inference`, {
        text: overrides?.text ?? text,
        interventions: overrides?.interventions ?? interventions,
        lens_type: overrides?.lensType ?? lensType,
        apply_chat_template: overrides?.useChatTemplate ?? useChatTemplate,
        return_attention: true // Always fetch attention
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
    setInterventions(prev => {
      const currentList = prev[layer] || [];

      // Remove existing intervention on the same token (or same "all tokens" setting)
      // For block_attention, we might want to allow multiple unless exact same source/target?
      // But user request specifically mentioned "same token". 
      // Let's apply strict replacement for stream interventions (scale/zero) based on token_index.

      let newList = currentList;

      if (type !== 'block_attention') {
        newList = currentList.filter(item => {
          // If both target specific token, match index
          if (item.token_index !== undefined && tokenIndex !== undefined) {
            return item.token_index !== tokenIndex;
          }
          // If both target all tokens (undefined index), match that
          if (item.token_index === undefined && tokenIndex === undefined) {
            return false; // Remove existing "all tokens" intervention
          }
          return true;
        });
      }

      const newConfig = { type, value, token_index: tokenIndex, source_tokens: sourceTokens, target_tokens: targetTokens, all_layers: allLayers };
      return {
        ...prev,
        [layer]: [...newList, newConfig]
      };
    });
  };

  const removeIntervention = (layer: string, index: number) => {
    setInterventions(prev => {
      const currentList = prev[layer] || [];
      const newList = currentList.filter((_, i) => i !== index);

      const next = { ...prev };
      if (newList.length === 0) {
        delete next[layer];
      } else {
        next[layer] = newList;
      }
      return next;
    });
  };

  const [sessions, setSessions] = useState<{ id: string, name: string, timestamp: string }[]>([]);
  const [showLoadModal, setShowLoadModal] = useState(false);
  const [loadVizOnly, setLoadVizOnly] = useState(false);
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
      setLoading(true);
      setLoadingStatus("Loading session data...");

      const res = await axios.post(`${API_URL}/load_session`, { session_id: sessionId });
      const json = res.data;

      // Restore state
      setText(json.text);

      // Handle legacy format (single config per layer)
      const restoredInterventions: Interventions = {};
      for (const [key, val] of Object.entries(json.interventions)) {
        if (Array.isArray(val)) {
          restoredInterventions[key] = val as any[];
        } else {
          restoredInterventions[key] = [val as any];
        }
      }
      setInterventions(restoredInterventions);

      setLensType(json.lens_type);
      setUseChatTemplate(json.apply_chat_template);

      if (loadVizOnly) {
        // Visualization Only Mode
        if (json.results) {
          setResults(json.results);
          alert("Loaded visualization only. Model was not switched.");
        } else {
          alert("No results found in saved session.");
        }
        setShowLoadModal(false);
        setLoading(false);
        setLoadingStatus("");
        return;
      }

      // Full Load Mode
      const needsModelSwitch = json.model_name && json.model_name !== selectedModel;
      const needsBackendSwitch = json.backend && json.backend !== selectedBackend;

      if (needsModelSwitch || needsBackendSwitch) {
        const modelToLoad = json.model_name || selectedModel;
        const backendToUse = json.backend || selectedBackend;

        setLoadingStatus(`Switching to ${backendToUse} backend with ${modelToLoad}...`);
        setSelectedModel(modelToLoad);
        setSelectedBackend(backendToUse);

        // Wait for model to load before running inference
        const success = await loadModel(modelToLoad, backendToUse);
        if (!success) {
          alert(`Failed to switch to model ${modelToLoad} with ${backendToUse} backend. Inference aborted.`);
          setLoading(false);
          setLoadingStatus("");
          return;
        }
      }

      // setShowLoadModal(false); // Don't close here, wait for inference

      // Auto-rerun inference with loaded data
      setLoadingStatus("Re-running inference...");
      await runInference({
        text: json.text,
        interventions: restoredInterventions,
        lensType: json.lens_type,
        useChatTemplate: json.apply_chat_template
      });

      setLoadingStatus("");
      setShowLoadModal(false); // Close modal only after everything is done
      setLoading(false);
    } catch (err) {
      console.error("Failed to load session", err);
      alert("Failed to load session");
      setLoading(false);
      setLoadingStatus("");
    }
  };

  const [tooltipData, setTooltipData] = useState<{
    layer: number;
    token: number;
    rect: DOMRect;
    preds: any[];
    layerName: string;
  } | null>(null);

  const deleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent loading the session
    if (!confirm("Are you sure you want to delete this session? This action cannot be undone.")) {
      return;
    }

    try {
      await axios.post(`${API_URL}/delete_session`, { session_id: sessionId });
      // Refresh list
      fetchSessions();
    } catch (err) {
      console.error("Failed to delete session", err);
      alert("Failed to delete session");
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>LogitLens Advanced</h1>
        <div className="header-controls flex-row" style={{ alignItems: 'center', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <label style={{ fontSize: '0.85em', fontWeight: 'bold' }}>Backend:</label>
            <select value={selectedBackend} onChange={handleBackendChange} disabled={loading} style={{ fontSize: '0.9em' }}>
              <option value="custom">Custom (Llama/Qwen3)</option>
              <option value="transformerlens">TransformerLens</option>
            </select>
          </div>

          {!showMoreModels ? (
            <select value={selectedModel} onChange={handleModelChange} disabled={loading}>
              <optgroup label="Llama Models">
                <option value="meta-llama/Llama-3.2-1B">Llama 3.2 1B</option>
                <option value="meta-llama/Llama-3.2-3B">Llama 3.2 3B</option>
                <option value="meta-llama/Llama-3.2-1B-Instruct">Llama 3.2 1B Instruct</option>
                <option value="meta-llama/Llama-3.2-3B-Instruct">Llama 3.2 3B Instruct</option>
                <option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama 1.1B</option>
              </optgroup>

              {selectedBackend === "custom" && (
                <optgroup label="Qwen3 Models (Custom Backend Only)">
                  <option value="Qwen/Qwen3-0.6B">Qwen3 0.6B (Instruct)</option>
                  <option value="Qwen/Qwen3-0.6B-Base">Qwen3 0.6B Base</option>
                  <option value="Qwen/Qwen3-1.7B">Qwen3 1.7B (Instruct)</option>
                  <option value="Qwen/Qwen3-1.7B-Base">Qwen3 1.7B Base</option>
                  <option value="Qwen/Qwen3-4B">Qwen3 4B (Instruct)</option>
                  <option value="Qwen/Qwen3-4B-Base">Qwen3 4B Base</option>
                </optgroup>
              )}

              {selectedBackend === "transformerlens" && (
                <>
                  <optgroup label="Qwen2.5 Models (TransformerLens)">
                    <option value="Qwen/Qwen2.5-0.5B">Qwen2.5 0.5B</option>
                    <option value="Qwen/Qwen2.5-0.5B-Instruct">Qwen2.5 0.5B Instruct</option>
                    <option value="Qwen/Qwen2.5-1.5B">Qwen2.5 1.5B</option>
                    <option value="Qwen/Qwen2.5-1.5B-Instruct">Qwen2.5 1.5B Instruct</option>
                    <option value="Qwen/Qwen2.5-3B">Qwen2.5 3B</option>
                    <option value="Qwen/Qwen2.5-3B-Instruct">Qwen2.5 3B Instruct</option>
                    <option value="Qwen/Qwen2.5-7B">Qwen2.5 7B</option>
                    <option value="Qwen/Qwen2.5-7B-Instruct">Qwen2.5 7B Instruct</option>
                  </optgroup>
                  <optgroup label="GPT-2 Models (TransformerLens)">
                    <option value="gpt2">GPT-2 (124M)</option>
                    <option value="gpt2-medium">GPT-2 Medium (355M)</option>
                    <option value="gpt2-large">GPT-2 Large (774M)</option>
                    <option value="gpt2-xl">GPT-2 XL (1.5B)</option>
                  </optgroup>
                  <optgroup label="Mistral Models (TransformerLens)">
                    <option value="mistralai/Mistral-7B-v0.1">Mistral 7B</option>
                    <option value="mistralai/Mistral-7B-Instruct-v0.1">Mistral 7B Instruct</option>
                  </optgroup>
                  <optgroup label="Phi Models (TransformerLens)">
                    <option value="microsoft/phi-2">Phi-2 (2.7B)</option>
                    <option value="microsoft/Phi-3-mini-4k-instruct">Phi-3 Mini 4K Instruct</option>
                  </optgroup>
                  <optgroup label="Gemma Models (TransformerLens)">
                    <option value="google/gemma-2b">Gemma 2B</option>
                    <option value="google/gemma-2b-it">Gemma 2B Instruct</option>
                    <option value="google/gemma-7b">Gemma 7B</option>
                    <option value="google/gemma-7b-it">Gemma 7B Instruct</option>
                  </optgroup>
                </>
              )}

              {selectedBackend === "transformerlens" && (
                <optgroup label="More Models">
                  <option value="__more_models__">Browse All TransformerLens Models (200+)...</option>
                </optgroup>
              )}
            </select>
          ) : (
            <SearchableModelDropdown
              value={selectedModel}
              onChange={handleMoreModelsSelect}
              disabled={loading}
              onClose={() => setShowMoreModels(false)}
            />
          )}

          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              fontSize: '0.9em',
              opacity: isInstructModel(selectedModel) ? 1 : 0.5,
              cursor: isInstructModel(selectedModel) ? 'pointer' : 'not-allowed'
            }}
            title={!isInstructModel(selectedModel) ? "This is a base model and should not be used with a chat template." : ""}
          >
            <input
              type="checkbox"
              checked={useChatTemplate}
              onChange={(e) => setUseChatTemplate(e.target.checked)}
              disabled={loading || !isInstructModel(selectedModel)}
            />
            Use Chat Template
          </label>

          <select value={lensType} onChange={e => setLensType(e.target.value)} disabled={loading}>
            <option value="block_output">Block Output (Standard)</option>
            <option value="post_attention">Post-Attention</option>
            <option value="combined">Combined (Most Detailed)</option>
          </select>



          <div className="status">
            Status: {loading ? (loadingStatus || "Loading...") : modelLoaded ? "Ready" : "Not Loaded"}
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

            {loading && (
              <div style={{ marginBottom: '15px', padding: '10px', backgroundColor: '#2a2a2a', borderRadius: '4px', border: '1px solid #444', color: '#4caf50', textAlign: 'center' }}>
                <div className="loading-spinner" style={{ display: 'inline-block', width: '12px', height: '12px', border: '2px solid #4caf50', borderTop: '2px solid transparent', borderRadius: '50%', animation: 'spin 1s linear infinite', marginRight: '8px' }}></div>
                <strong>{loadingStatus}</strong>
                <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
              </div>
            )}

            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '15px', padding: '8px', backgroundColor: '#333', borderRadius: '4px' }}>
              <input
                type="checkbox"
                checked={loadVizOnly}
                onChange={e => setLoadVizOnly(e.target.checked)}
              />
              <span style={{ fontSize: '0.9em' }}>
                <strong>Load Visualization Only</strong>
                <br />
                <span style={{ fontSize: '0.85em', color: '#aaa' }}>Skip model switch & inference (fast)</span>
              </span>
            </label>

            <div className="session-list" style={{ display: 'flex', flexDirection: 'column', gap: '8px', margin: '15px 0' }}>
              {sessions.length === 0 ? (
                <p style={{ color: '#888' }}>No saved sessions found.</p>
              ) : (
                sessions.map(s => (
                  <div key={s.id} style={{ display: 'flex', gap: '5px' }}>
                    <button
                      onClick={() => loadSession(s.id)}
                      className="secondary-btn"
                      disabled={loading}
                      style={{ flex: 1, textAlign: 'left', padding: '10px', display: 'flex', justifyContent: 'space-between', opacity: loading ? 0.5 : 1, cursor: loading ? 'not-allowed' : 'pointer' }}
                    >
                      <span>{s.name}</span>
                      <span style={{ fontSize: '0.8em', color: '#888' }}>{s.timestamp}</span>
                    </button>
                    <button
                      onClick={(e) => deleteSession(s.id, e)}
                      className="secondary-btn"
                      disabled={loading}
                      title="Delete Session"
                      style={{ padding: '0 12px', color: '#ff4444', borderColor: '#552222', opacity: loading ? 0.5 : 1, cursor: loading ? 'not-allowed' : 'pointer' }}
                    >
                      🗑️
                    </button>
                  </div>
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
              {Object.entries(interventions).map(([layer, configs]) => (
                <div key={layer} style={{ marginBottom: '10px', border: '1px solid #333', padding: '5px', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '5px', fontSize: '0.9em', color: '#aaa' }}>{layer}</div>
                  {configs.map((config, idx) => (
                    <div key={idx} className="intervention-item flex-row" style={{ alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px', backgroundColor: '#222', padding: '4px', borderRadius: '2px' }}>
                      <span style={{ fontSize: '0.9em' }}>
                        {config.type}
                        {config.type === 'scale' && `(${config.value})`}
                        {config.type === 'block_attention' && config.source_tokens && config.target_tokens && ` (${config.source_tokens.join(',')} → ${config.target_tokens.join(',')})`}
                        {config.token_index != null && ` @ Token ${config.token_index}`}
                        {config.all_layers && ` [ALL LAYERS]`}
                      </span>
                      <button onClick={() => removeIntervention(layer, idx)} style={{ padding: '2px 6px', fontSize: '0.8em', marginLeft: '10px' }}>X</button>
                    </div>
                  ))}
                </div>
              ))}

              {/* Stream Interventions Section */}
              <h4 style={{ marginTop: '1.5rem', marginBottom: '0.5rem' }}>Stream Interventions</h4>
              <p style={{ fontSize: '0.85em', color: '#888', marginBottom: '0.5rem' }}>
                Modify activation streams (zero out, scale, etc.)
              </p>
              <div className="add-intervention flex-row" style={{ marginTop: '0.5rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <select id="stream-layer-select" disabled={streamAllLayers}>
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
                <input type="number" id="stream-token-input" placeholder="Token Idx (opt)" style={{ width: '110px' }} />
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <input
                    type="checkbox"
                    id="stream-all-layers"
                    checked={streamAllLayers}
                    onChange={(e) => setStreamAllLayers(e.target.checked)}
                  />
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

            <button onClick={() => runInference()} disabled={loading || !modelLoaded} className="primary-btn">
              Run LogitLens
            </button>
          </div>
        </div>

        <div className="visualization card">
          <h2>LogitLens Visualization</h2>
          {results ? (
            <div className="logit-lens-view" style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', width: '100%' }}>
                <thead>
                  {/* Input tokens row */}
                  <tr style={{ backgroundColor: 'rgba(0,0,0,0.3)' }}>
                    <th style={{
                      textAlign: 'left',
                      padding: '6px 8px',
                      borderBottom: '1px solid #444',
                      minWidth: '120px',
                      position: 'sticky',
                      left: 0,
                      background: 'rgba(30,30,30,0.95)',
                      zIndex: 1,
                      fontSize: '0.75em',
                      color: '#888',
                      fontWeight: 'normal'
                    }}>Input</th>
                    {results.input_tokens?.map((token, i) => (
                      <th key={i} style={{
                        padding: '6px 8px',
                        borderBottom: '1px solid #444',
                        minWidth: '80px',
                        textAlign: 'center',
                        fontSize: '0.85em',
                        color: '#ccc',
                        fontWeight: 'normal',
                        fontFamily: 'monospace'
                      }} title={`Token ${i}: "${token}"`}>
                        {formatToken(token)}
                      </th>
                    ))}
                  </tr>
                  {/* Position index row */}
                  <tr style={{ backgroundColor: 'rgba(0,0,0,0.2)' }}>
                    <th style={{
                      textAlign: 'left',
                      padding: '6px 8px',
                      borderBottom: '1px solid #444',
                      minWidth: '120px',
                      position: 'sticky',
                      left: 0,
                      background: 'rgba(30,30,30,0.95)',
                      zIndex: 1,
                      fontSize: '0.75em',
                      color: '#666',
                      fontWeight: 'normal'
                    }}>Layer / Position →</th>
                    {results.logit_lens[0]?.predictions.map((_, i) => (
                      <th key={i} style={{
                        padding: '6px 8px',
                        borderBottom: '1px solid #444',
                        minWidth: '80px',
                        textAlign: 'center',
                        fontSize: '0.75em',
                        color: '#666',
                        fontWeight: 'normal'
                      }}>
                        [{i}]
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

        {results && (
          <div className="attention-viz card" style={{ gridColumn: '1 / -1' }}>
            <h2>Attention Visualization</h2>

            <div className="attention-controls flex-row" style={{ gap: '1rem', alignItems: 'center', marginBottom: '10px', flexWrap: 'wrap' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.9em', fontWeight: 'bold' }}>
                <input
                  type="checkbox"
                  checked={showAttention}
                  onChange={(e) => setShowAttention(e.target.checked)}
                  disabled={loading}
                />
                Show Attention
              </label>

              {showAttention && (
                <>
                  <select value={attnAggregation} onChange={e => setAttnAggregation(e.target.value)} disabled={loading}>
                    <option value="all_average">Average All</option>
                    <option value="layer_mean">Layer Mean</option>
                    <option value="layer_mean_grid">Layer Mean (Grid)</option>
                    <option value="specific">Specific Head</option>
                    <option value="specific_grid">Specific Head (Grid)</option>
                  </select>

                  {(attnAggregation === "specific" || attnAggregation === "layer_mean" || attnAggregation === "specific_grid") && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                      <span style={{ fontSize: '0.8em' }}>Layer:</span>
                      <input
                        type="range"
                        min="0"
                        max={results && results.attention && results.attention.length > 0 ? results.attention.length - 1 : 31}
                        value={attnLayer}
                        onChange={e => setAttnLayer(parseInt(e.target.value))}
                        disabled={loading}
                      />
                      <span style={{ fontSize: '0.8em', minWidth: '20px' }}>{attnLayer}</span>
                    </div>
                  )}

                  {attnAggregation === "specific" && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                      <span style={{ fontSize: '0.8em' }}>Head:</span>
                      <input
                        type="range"
                        min="0"
                        max={results && results.attention && results.attention[0] && results.attention[0].length > 0 ? results.attention[0].length - 1 : 31}
                        value={attnHead}
                        onChange={e => setAttnHead(parseInt(e.target.value))}
                        disabled={loading}
                      />
                      <span style={{ fontSize: '0.8em', minWidth: '20px' }}>{attnHead}</span>
                    </div>
                  )}
                </>
              )}
            </div>

            {showAttention && (
              <>
                {!results.attention ? (
                  <div style={{ padding: '20px', textAlign: 'center', color: '#888', backgroundColor: '#222', borderRadius: '4px' }}>
                    <p>Attention data not available in current results.</p>
                    <button onClick={() => runInference()} disabled={loading} className="primary-btn" style={{ marginTop: '10px' }}>
                      Re-run Inference to Fetch Attention
                    </button>
                  </div>
                ) : (
                  <>
                    <p style={{ fontSize: '0.9em', color: '#888', marginBottom: '10px' }}>
                      {attnAggregation === 'all_average' ? '' :
                        attnAggregation === 'layer_mean' ? `` :
                          attnAggregation === 'layer_mean_grid' ? 'Mean of all heads for each layer' :
                            attnAggregation === 'specific_grid' ? `Layer ${attnLayer} (All Heads)` :
                              ``}
                    </p>

                    {(() => {
                      // Use actual input tokens instead of embedding predictions, with formatting for display
                      const rawTokens = results.input_tokens || results.logit_lens[0].predictions.map((p: any) => p[0].token);
                      const tokens = rawTokens.map(formatToken);
                      const attnData = results.attention; // [layers, heads, seq, seq]

                      if (!attnData || !attnData.length) return <div>No attention data</div>;

                      if (attnAggregation === 'all_average') {
                        // Compute average across all layers and heads
                        const numLayers = attnData.length;
                        const numHeads = attnData[0].length;
                        const seqLen = attnData[0][0].length;

                        // Initialize zero matrix
                        const avgMatrix = Array(seqLen).fill(0).map(() => Array(seqLen).fill(0));

                        for (let l = 0; l < numLayers; l++) {
                          for (let h = 0; h < numHeads; h++) {
                            for (let i = 0; i < seqLen; i++) {
                              for (let j = 0; j < seqLen; j++) {
                                avgMatrix[i][j] += attnData[l][h][i][j];
                              }
                            }
                          }
                        }

                        // Divide by total count
                        const total = numLayers * numHeads;
                        for (let i = 0; i < seqLen; i++) {
                          for (let j = 0; j < seqLen; j++) {
                            avgMatrix[i][j] /= total;
                          }
                        }

                        const saveName = `${selectedModel.split('/').pop()}_${text.slice(0, 10).replace(/\s+/g, '_')}_AvgAll`;
                        return <AttentionHeatmap data={avgMatrix} tokens={tokens} title="Average Attention (All Layers & Heads)" saveName={saveName} />;
                      }

                      if (attnAggregation === 'layer_mean') {
                        // Compute mean for specific layer
                        const layerData = attnData[attnLayer]; // [heads, seq, seq]
                        if (!layerData) return <div>Invalid layer</div>;

                        const numHeads = layerData.length;
                        const seqLen = layerData[0].length;
                        const layerMeanMatrix = Array(seqLen).fill(0).map(() => Array(seqLen).fill(0));

                        for (let h = 0; h < numHeads; h++) {
                          for (let i = 0; i < seqLen; i++) {
                            for (let j = 0; j < seqLen; j++) {
                              layerMeanMatrix[i][j] += layerData[h][i][j];
                            }
                          }
                        }

                        // Divide by num heads
                        for (let i = 0; i < seqLen; i++) {
                          for (let j = 0; j < seqLen; j++) {
                            layerMeanMatrix[i][j] /= numHeads;
                          }
                        }

                        const saveName = `${selectedModel.split('/').pop()}_${text.slice(0, 10).replace(/\s+/g, '_')}_L${attnLayer}_Mean`;
                        return <AttentionHeatmap data={layerMeanMatrix} tokens={tokens} title={`Layer ${attnLayer} Mean Attention`} saveName={saveName} />;
                      }

                      if (attnAggregation === 'specific') {
                        const matrix = attnData[attnLayer][attnHead];
                        if (!matrix) return <div>Invalid head</div>;
                        const saveName = `${selectedModel.split('/').pop()}_${text.slice(0, 10).replace(/\s+/g, '_')}_L${attnLayer}_H${attnHead}`;
                        return <AttentionHeatmap data={matrix} tokens={tokens} title={`Layer ${attnLayer}, Head ${attnHead}`} saveName={saveName} />;
                      }

                      if (attnAggregation === 'specific_grid') {
                        const layerData = attnData[attnLayer]; // [heads, seq, seq]
                        if (!layerData) return <div>Invalid layer</div>;

                        const handleSaveGrid = async () => {
                          try {
                            const res = await axios.post(`${API_URL}/save_grid_visualization`, {
                              grid_data: layerData,
                              tokens: tokens,
                              title: `${selectedModel.split('/').pop()}_${text.slice(0, 10).replace(/\s+/g, '_')}_L${attnLayer}_AllHeads`,
                              grid_type: "head_grid"
                            });
                            alert(`Grid visualization saved to: ${res.data.filepath}`);
                          } catch (err) {
                            console.error("Failed to save grid visualization", err);
                            alert("Failed to save grid visualization");
                          }
                        };

                        return (
                          <div>
                            <div style={{ marginBottom: '10px' }}>
                              <button onClick={handleSaveGrid} className="save-btn">
                                Save All Heads
                              </button>
                            </div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                              {layerData.map((headData: number[][], h: number) => (
                                <AttentionHeatmap
                                  key={h}
                                  data={headData}
                                  tokens={tokens}
                                  title={`Head ${h}`}
                                  size={15} // Smaller size for grid
                                />
                              ))}
                            </div>
                          </div>
                        );
                      }

                      if (attnAggregation === 'layer_mean_grid') {
                        // Compute mean for ALL layers
                        const numLayers = attnData.length;
                        const numHeads = attnData[0].length;
                        const seqLen = attnData[0][0].length;

                        // Pre-compute means for all layers
                        const layerMeans: number[][][] = [];
                        for (let l = 0; l < numLayers; l++) {
                          const avgMatrix = Array(seqLen).fill(0).map(() => Array(seqLen).fill(0));
                          for (let h = 0; h < numHeads; h++) {
                            for (let i = 0; i < seqLen; i++) {
                              for (let j = 0; j < seqLen; j++) {
                                avgMatrix[i][j] += attnData[l][h][i][j];
                              }
                            }
                          }
                          for (let i = 0; i < seqLen; i++) {
                            for (let j = 0; j < seqLen; j++) {
                              avgMatrix[i][j] /= numHeads;
                            }
                          }
                          layerMeans.push(avgMatrix);
                        }

                        const handleSaveGrid = async () => {
                          try {
                            const res = await axios.post(`${API_URL}/save_grid_visualization`, {
                              grid_data: layerMeans,
                              tokens: tokens,
                              title: `${selectedModel.split('/').pop()}_${text.slice(0, 10).replace(/\s+/g, '_')}_AllLayers_Mean`,
                              grid_type: "layer_grid"
                            });
                            alert(`Grid visualization saved to: ${res.data.filepath}`);
                          } catch (err) {
                            console.error("Failed to save grid visualization", err);
                            alert("Failed to save grid visualization");
                          }
                        };

                        return (
                          <div>
                            <div style={{ marginBottom: '10px' }}>
                              <button onClick={handleSaveGrid} className="save-btn">
                                Save All Layers
                              </button>
                            </div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                              {layerMeans.map((layerData: number[][], l: number) => (
                                <AttentionHeatmap
                                  key={l}
                                  data={layerData}
                                  tokens={tokens}
                                  title={`Layer ${l}`}
                                  size={15} // Smaller size for grid
                                />
                              ))}
                            </div>
                          </div>
                        );
                      }

                      return null;
                    })()}
                  </>
                )}
              </>
            )}
          </div>
        )}
      </main>


      {/* Global Fixed Tooltip */}
      {
        tooltipData && (() => {
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
        })()
      }
    </div >
  )
}

export default App
