"""
Updated server with support for both custom and TransformerLens backends.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import threading
import datetime
import json
import shutil

# Import backends
from backends.custom_backend import CustomBackend
from backends.transformerlens_backend import TransformerLensBackend
from backends.base import BaseBackend

# Visualization imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
current_backend: Optional[BaseBackend] = None
backend_type: str = "custom"  # "custom" or "transformerlens"
model_lock = threading.Lock()
last_run_state = {}

# Pydantic models
class LoadModelRequest(BaseModel):
    model_name: str = "meta-llama/Llama-3.2-1B"
    backend: str = "custom"  # "custom" or "transformerlens"

class InterventionConfig(BaseModel):
    type: str
    value: Optional[float] = None
    vector: Optional[List[float]] = None
    token_index: Optional[int] = None
    source_tokens: Optional[List[int]] = None
    target_tokens: Optional[List[int]] = None
    all_layers: Optional[bool] = None

class InferenceRequest(BaseModel):
    text: str
    interventions: Dict[str, List[InterventionConfig]] = {}
    lens_type: str = "block_output"
    apply_chat_template: bool = False
    return_attention: bool = False

class SaveStateRequest(BaseModel):
    name: Optional[str] = None

class LoadSessionRequest(BaseModel):
    session_id: str

class DeleteSessionRequest(BaseModel):
    session_id: str

class SaveVisualizationRequest(BaseModel):
    attention_data: List[List[float]]
    tokens: List[str]
    title: str

class SaveGridVisualizationRequest(BaseModel):
    grid_data: List[List[List[float]]]
    tokens: List[str]
    title: str
    grid_type: str


def get_backend(backend_name: str) -> BaseBackend:
    """Get or create a backend instance."""
    if backend_name == "transformerlens":
        return TransformerLensBackend()
    else:
        return CustomBackend()


@app.post("/load_model")
def load_model(req: LoadModelRequest):
    global current_backend, backend_type

    # Quick check before lock
    if (current_backend is not None and
        current_backend.is_loaded() and
        backend_type == req.backend):
        info = current_backend.get_model_info()
        if info.get("model_name") == req.model_name:
            return {"status": "already_loaded", **info}

    with model_lock:
        # Check again inside lock
        if (current_backend is not None and
            current_backend.is_loaded() and
            backend_type == req.backend):
            info = current_backend.get_model_info()
            if info.get("model_name") == req.model_name:
                return {"status": "already_loaded", **info}

        try:
            # Unload previous backend if exists
            if current_backend is not None:
                current_backend.unload_model()

            # Create new backend
            print(f"Loading model with backend: {req.backend}")
            current_backend = get_backend(req.backend)
            backend_type = req.backend

            # Load model
            result = current_backend.load_model(req.model_name, device="cpu")
            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_status")
def get_model_status():
    global current_backend, backend_type

    if current_backend is None or not current_backend.is_loaded():
        return {"loaded": False, "backend": backend_type}

    info = current_backend.get_model_info()
    info["backend"] = backend_type
    return info


@app.post("/inference")
def inference(req: InferenceRequest):
    global current_backend, last_run_state

    if current_backend is None or not current_backend.is_loaded():
        raise HTTPException(status_code=400, detail="Model not loaded")

    try:
        # Run inference through backend
        result = current_backend.run_inference(
            text=req.text,
            interventions=req.interventions,
            lens_type=req.lens_type,
            apply_chat_template=req.apply_chat_template,
            return_attention=req.return_attention
        )

        # Store state for saving
        last_run_state = {
            "config": {
                "text": req.text,
                "interventions": {
                    k: [v.dict() for v in val] if isinstance(val, list) else val.dict()
                    for k, val in req.interventions.items()
                },
                "lens_type": req.lens_type,
                "model_name": current_backend.get_model_info().get("model_name"),
                "backend": backend_type,
                "apply_chat_template": req.apply_chat_template,
                "results": {
                    "text": req.text,
                    "logit_lens": result["logit_lens"]
                }
            },
            "tensors": result["tensors"]
        }

        # Return response (without tensors)
        response = {
            "text": result["text"],
            "input_tokens": result["input_tokens"],
            "logit_lens": result["logit_lens"]
        }

        if "attention" in result:
            response["attention"] = result["attention"]

        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Session management
SAVED_STATES_DIR = "saved_states"
if not os.path.exists(SAVED_STATES_DIR):
    os.makedirs(SAVED_STATES_DIR)


@app.post("/save_state")
def save_state(req: SaveStateRequest):
    if not last_run_state:
        raise HTTPException(status_code=400, detail="No state to save")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name_part = f"_{req.name}" if req.name else ""
    session_id = f"{timestamp}{name_part}"

    session_dir = os.path.join(SAVED_STATES_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Save config JSON
    config_path = os.path.join(session_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(last_run_state["config"], f, indent=2)

    # Save tensors NPZ
    tensors_path = os.path.join(session_dir, "tensors.npz")
    np.savez(tensors_path, **last_run_state["tensors"])

    return {"status": "saved", "session_id": session_id, "path": session_dir}


@app.get("/sessions")
def list_sessions():
    if not os.path.exists(SAVED_STATES_DIR):
        return []

    sessions = []
    for name in sorted(os.listdir(SAVED_STATES_DIR), reverse=True):
        path = os.path.join(SAVED_STATES_DIR, name)
        if os.path.isdir(path):
            try:
                parts = name.split("_", 2)
                timestamp_str = f"{parts[0]}_{parts[1]}"
                display_name = parts[2] if len(parts) > 2 else "Untitled"
            except:
                timestamp_str = "Unknown"
                display_name = name

            sessions.append({
                "id": name,
                "name": display_name,
                "timestamp": timestamp_str
            })
    return sessions


@app.post("/delete_session")
def delete_session(req: DeleteSessionRequest):
    session_dir = os.path.join(SAVED_STATES_DIR, req.session_id)

    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        shutil.rmtree(session_dir)
        return {"status": "deleted", "session_id": req.session_id}
    except Exception as e:
        print(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@app.post("/load_session")
def load_session(req: LoadSessionRequest):
    global last_run_state
    session_dir = os.path.join(SAVED_STATES_DIR, req.session_id)

    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")

    # Load config
    config_path = os.path.join(session_dir, "config.json")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Config file not found")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Ensure legacy compatibility
    if "apply_chat_template" not in config:
        config["apply_chat_template"] = False
    if "backend" not in config:
        config["backend"] = "custom"

    # Load tensors
    tensors_path = os.path.join(session_dir, "tensors.npz")
    tensors = {}
    if os.path.exists(tensors_path):
        loaded = np.load(tensors_path, allow_pickle=True)
        tensors = {k: loaded[k] for k in loaded.files}

    # Restore state
    last_run_state = {
        "config": config,
        "tensors": tensors
    }

    return config


# Visualization endpoints
@app.post("/save_visualization")
def save_visualization(req: SaveVisualizationRequest):
    viz_dir = "attention_visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join([c if c.isalnum() else "_" for c in req.title])
    filename = f"{timestamp}_{safe_title}.png"
    filepath = os.path.join(viz_dir, filename)

    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(req.attention_data, xticklabels=req.tokens, yticklabels=req.tokens, cmap="viridis")
        plt.title(req.title)
        plt.xlabel("Key Token")
        plt.ylabel("Query Token")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

        return {"status": "success", "filepath": filepath}
    except Exception as e:
        print(f"Error saving visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_grid_visualization")
def save_grid_visualization(req: SaveGridVisualizationRequest):
    import math

    viz_dir = "attention_visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join([c if c.isalnum() else "_" for c in req.title])
    filename = f"{timestamp}_{safe_title}.png"
    filepath = os.path.join(viz_dir, filename)

    try:
        num_plots = len(req.grid_data)
        cols = 4
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten() if num_plots > 1 else [axes]

        for i, ax in enumerate(axes):
            if i < num_plots:
                sns.heatmap(req.grid_data[i], xticklabels=req.tokens, yticklabels=req.tokens,
                          cmap="viridis", ax=ax, cbar=False)
                sub_title = f"Layer {i}" if req.grid_type == "layer_grid" else f"Head {i}"
                ax.set_title(sub_title)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.tick_params(axis='y', rotation=0, labelsize=8)
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

        return {"status": "success", "filepath": filepath}
    except Exception as e:
        print(f"Error saving grid visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
