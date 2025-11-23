# Logit Lens Llama Advanced

An interactive tool to analyze and intervene on Llama model internal states using Logit Lens.


## Demo

![demo](banner.gif)

## Features

- **Interactive Logit Lens**: Visualize the model's prediction at every layer (Embeddings -> Layers -> Output).
- **Interventions**: Modify internal streams (Residual, Attention, MLP) in real-time.
    - **Zero**: Zero out specific vectors.
    - **Scale**: Scale vectors by a factor.
    - **Block Attention**: Prevent information flow between specific tokens or across layers.
        - FYI: Attention blocking mechanism happens before softmax, sending -inf to attention scores. Attention scores will add up to 1.
- **Session Management**: Save and load your analysis sessions (config + full tensor state).

## Installation

1.  **Backend**:
    ```bash
    uv sync
    ```

2.  **Frontend**:
    ```bash
    cd frontend
    npm install
    ```
In order to access Llama models from meta-llama you would need to set up a [huggingface token](https://huggingface.co/docs/hub/en/security-tokens). 

## Usage

1.  **Start the Backend**:
    ```bash
    uv run uvicorn server:app --reload --port 8000
    ```

2.  **Start the Frontend**:
    ```bash
    cd frontend
    npm run dev
    ```

3.  Open `http://localhost:5173` in your browser.

## Architecture

- `model.py`: Custom `LlamaModel` with hooks for interventions.
- `logit_lens.py`: Utilities for projecting hidden states to vocabulary.
- `server.py`: FastAPI backend handling inference and state management.
- `frontend/`: React + Vite application.

## Interventions

The application supports modifying the following streams:
- `embeddings`: The initial token embeddings.
- `layer_X_attn_output`: The output of the Attention mechanism at layer X.
- `layer_X_mlp_output`: The output of the MLP at layer X.
- `layer_X_output`: The final output of layer X (after residual connection).

## License

MIT
