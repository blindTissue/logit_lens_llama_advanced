/**
 * Official TransformerLens supported models from the library.
 * Source: transformer_lens/loading_from_pretrained.py
 */

export interface ModelInfo {
  id: string;
  name: string;
  family: string;
  size?: string;
}

export const TRANSFORMERLENS_MODELS: ModelInfo[] = [
  // GPT-2 Models
  { id: "gpt2", name: "GPT-2 (124M)", family: "GPT-2", size: "124M" },
  { id: "gpt2-medium", name: "GPT-2 Medium (355M)", family: "GPT-2", size: "355M" },
  { id: "gpt2-large", name: "GPT-2 Large (774M)", family: "GPT-2", size: "774M" },
  { id: "gpt2-xl", name: "GPT-2 XL (1.5B)", family: "GPT-2", size: "1.5B" },
  { id: "distilgpt2", name: "DistilGPT-2", family: "GPT-2", size: "82M" },

  // OPT Models
  { id: "facebook/opt-125m", name: "OPT 125M", family: "OPT", size: "125M" },
  { id: "facebook/opt-1.3b", name: "OPT 1.3B", family: "OPT", size: "1.3B" },
  { id: "facebook/opt-2.7b", name: "OPT 2.7B", family: "OPT", size: "2.7B" },
  { id: "facebook/opt-6.7b", name: "OPT 6.7B", family: "OPT", size: "6.7B" },
  { id: "facebook/opt-13b", name: "OPT 13B", family: "OPT", size: "13B" },
  { id: "facebook/opt-30b", name: "OPT 30B", family: "OPT", size: "30B" },
  { id: "facebook/opt-66b", name: "OPT 66B", family: "OPT", size: "66B" },

  // GPT-Neo/NeoX Models
  { id: "EleutherAI/gpt-neo-125M", name: "GPT-Neo 125M", family: "GPT-Neo", size: "125M" },
  { id: "EleutherAI/gpt-neo-1.3B", name: "GPT-Neo 1.3B", family: "GPT-Neo", size: "1.3B" },
  { id: "EleutherAI/gpt-neo-2.7B", name: "GPT-Neo 2.7B", family: "GPT-Neo", size: "2.7B" },
  { id: "EleutherAI/gpt-j-6B", name: "GPT-J 6B", family: "GPT-J", size: "6B" },
  { id: "EleutherAI/gpt-neox-20b", name: "GPT-NeoX 20B", family: "GPT-NeoX", size: "20B" },

  // Pythia Models
  { id: "EleutherAI/pythia-14m", name: "Pythia 14M", family: "Pythia", size: "14M" },
  { id: "EleutherAI/pythia-31m", name: "Pythia 31M", family: "Pythia", size: "31M" },
  { id: "EleutherAI/pythia-70m", name: "Pythia 70M", family: "Pythia", size: "70M" },
  { id: "EleutherAI/pythia-160m", name: "Pythia 160M", family: "Pythia", size: "160M" },
  { id: "EleutherAI/pythia-410m", name: "Pythia 410M", family: "Pythia", size: "410M" },
  { id: "EleutherAI/pythia-1b", name: "Pythia 1B", family: "Pythia", size: "1B" },
  { id: "EleutherAI/pythia-1.4b", name: "Pythia 1.4B", family: "Pythia", size: "1.4B" },
  { id: "EleutherAI/pythia-2.8b", name: "Pythia 2.8B", family: "Pythia", size: "2.8B" },
  { id: "EleutherAI/pythia-6.9b", name: "Pythia 6.9B", family: "Pythia", size: "6.9B" },
  { id: "EleutherAI/pythia-12b", name: "Pythia 12B", family: "Pythia", size: "12B" },
  { id: "EleutherAI/pythia-70m-deduped", name: "Pythia 70M (Deduped)", family: "Pythia", size: "70M" },
  { id: "EleutherAI/pythia-160m-deduped", name: "Pythia 160M (Deduped)", family: "Pythia", size: "160M" },
  { id: "EleutherAI/pythia-410m-deduped", name: "Pythia 410M (Deduped)", family: "Pythia", size: "410M" },
  { id: "EleutherAI/pythia-1b-deduped", name: "Pythia 1B (Deduped)", family: "Pythia", size: "1B" },
  { id: "EleutherAI/pythia-1.4b-deduped", name: "Pythia 1.4B (Deduped)", family: "Pythia", size: "1.4B" },
  { id: "EleutherAI/pythia-2.8b-deduped", name: "Pythia 2.8B (Deduped)", family: "Pythia", size: "2.8B" },
  { id: "EleutherAI/pythia-6.9b-deduped", name: "Pythia 6.9B (Deduped)", family: "Pythia", size: "6.9B" },
  { id: "EleutherAI/pythia-12b-deduped", name: "Pythia 12B (Deduped)", family: "Pythia", size: "12B" },

  // Llama Models
  { id: "meta-llama/Llama-2-7b-hf", name: "Llama 2 7B", family: "Llama", size: "7B" },
  { id: "meta-llama/Llama-2-7b-chat-hf", name: "Llama 2 7B Chat", family: "Llama", size: "7B" },
  { id: "meta-llama/Llama-2-13b-hf", name: "Llama 2 13B", family: "Llama", size: "13B" },
  { id: "meta-llama/Llama-2-13b-chat-hf", name: "Llama 2 13B Chat", family: "Llama", size: "13B" },
  { id: "meta-llama/Llama-2-70b-chat-hf", name: "Llama 2 70B Chat", family: "Llama", size: "70B" },
  { id: "codellama/CodeLlama-7b-hf", name: "CodeLlama 7B", family: "Llama", size: "7B" },
  { id: "codellama/CodeLlama-7b-Python-hf", name: "CodeLlama 7B Python", family: "Llama", size: "7B" },
  { id: "codellama/CodeLlama-7b-Instruct-hf", name: "CodeLlama 7B Instruct", family: "Llama", size: "7B" },
  { id: "meta-llama/Meta-Llama-3-8B", name: "Llama 3 8B", family: "Llama", size: "8B" },
  { id: "meta-llama/Meta-Llama-3-8B-Instruct", name: "Llama 3 8B Instruct", family: "Llama", size: "8B" },
  { id: "meta-llama/Meta-Llama-3-70B", name: "Llama 3 70B", family: "Llama", size: "70B" },
  { id: "meta-llama/Meta-Llama-3-70B-Instruct", name: "Llama 3 70B Instruct", family: "Llama", size: "70B" },
  { id: "meta-llama/Llama-3.1-8B", name: "Llama 3.1 8B", family: "Llama", size: "8B" },
  { id: "meta-llama/Llama-3.1-8B-Instruct", name: "Llama 3.1 8B Instruct", family: "Llama", size: "8B" },
  { id: "meta-llama/Llama-3.1-70B", name: "Llama 3.1 70B", family: "Llama", size: "70B" },
  { id: "meta-llama/Llama-3.1-70B-Instruct", name: "Llama 3.1 70B Instruct", family: "Llama", size: "70B" },
  { id: "meta-llama/Llama-3.2-1B", name: "Llama 3.2 1B", family: "Llama", size: "1B" },
  { id: "meta-llama/Llama-3.2-3B", name: "Llama 3.2 3B", family: "Llama", size: "3B" },
  { id: "meta-llama/Llama-3.2-1B-Instruct", name: "Llama 3.2 1B Instruct", family: "Llama", size: "1B" },
  { id: "meta-llama/Llama-3.2-3B-Instruct", name: "Llama 3.2 3B Instruct", family: "Llama", size: "3B" },
  { id: "meta-llama/Llama-3.3-70B-Instruct", name: "Llama 3.3 70B Instruct", family: "Llama", size: "70B" },

  // Mistral/Mixtral Models
  { id: "mistralai/Mistral-7B-v0.1", name: "Mistral 7B", family: "Mistral", size: "7B" },
  { id: "mistralai/Mistral-7B-Instruct-v0.1", name: "Mistral 7B Instruct", family: "Mistral", size: "7B" },
  { id: "mistralai/Mistral-Small-24B-Base-2501", name: "Mistral Small 24B Base", family: "Mistral", size: "24B" },
  { id: "mistralai/Mistral-Nemo-Base-2407", name: "Mistral Nemo Base", family: "Mistral", size: "12B" },
  { id: "mistralai/Mixtral-8x7B-v0.1", name: "Mixtral 8x7B", family: "Mixtral", size: "8x7B" },
  { id: "mistralai/Mixtral-8x7B-Instruct-v0.1", name: "Mixtral 8x7B Instruct", family: "Mixtral", size: "8x7B" },

  // Qwen Models
  { id: "Qwen/Qwen-1_8B", name: "Qwen 1.8B", family: "Qwen", size: "1.8B" },
  { id: "Qwen/Qwen-7B", name: "Qwen 7B", family: "Qwen", size: "7B" },
  { id: "Qwen/Qwen-14B", name: "Qwen 14B", family: "Qwen", size: "14B" },
  { id: "Qwen/Qwen-1_8B-Chat", name: "Qwen 1.8B Chat", family: "Qwen", size: "1.8B" },
  { id: "Qwen/Qwen-7B-Chat", name: "Qwen 7B Chat", family: "Qwen", size: "7B" },
  { id: "Qwen/Qwen-14B-Chat", name: "Qwen 14B Chat", family: "Qwen", size: "14B" },
  { id: "Qwen/Qwen1.5-0.5B", name: "Qwen1.5 0.5B", family: "Qwen", size: "0.5B" },
  { id: "Qwen/Qwen1.5-0.5B-Chat", name: "Qwen1.5 0.5B Chat", family: "Qwen", size: "0.5B" },
  { id: "Qwen/Qwen1.5-1.8B", name: "Qwen1.5 1.8B", family: "Qwen", size: "1.8B" },
  { id: "Qwen/Qwen1.5-1.8B-Chat", name: "Qwen1.5 1.8B Chat", family: "Qwen", size: "1.8B" },
  { id: "Qwen/Qwen1.5-4B", name: "Qwen1.5 4B", family: "Qwen", size: "4B" },
  { id: "Qwen/Qwen1.5-4B-Chat", name: "Qwen1.5 4B Chat", family: "Qwen", size: "4B" },
  { id: "Qwen/Qwen1.5-7B", name: "Qwen1.5 7B", family: "Qwen", size: "7B" },
  { id: "Qwen/Qwen1.5-7B-Chat", name: "Qwen1.5 7B Chat", family: "Qwen", size: "7B" },
  { id: "Qwen/Qwen1.5-14B", name: "Qwen1.5 14B", family: "Qwen", size: "14B" },
  { id: "Qwen/Qwen1.5-14B-Chat", name: "Qwen1.5 14B Chat", family: "Qwen", size: "14B" },
  { id: "Qwen/Qwen2-0.5B", name: "Qwen2 0.5B", family: "Qwen", size: "0.5B" },
  { id: "Qwen/Qwen2-0.5B-Instruct", name: "Qwen2 0.5B Instruct", family: "Qwen", size: "0.5B" },
  { id: "Qwen/Qwen2-1.5B", name: "Qwen2 1.5B", family: "Qwen", size: "1.5B" },
  { id: "Qwen/Qwen2-1.5B-Instruct", name: "Qwen2 1.5B Instruct", family: "Qwen", size: "1.5B" },
  { id: "Qwen/Qwen2-7B", name: "Qwen2 7B", family: "Qwen", size: "7B" },
  { id: "Qwen/Qwen2-7B-Instruct", name: "Qwen2 7B Instruct", family: "Qwen", size: "7B" },
  { id: "Qwen/Qwen2.5-0.5B", name: "Qwen2.5 0.5B", family: "Qwen", size: "0.5B" },
  { id: "Qwen/Qwen2.5-0.5B-Instruct", name: "Qwen2.5 0.5B Instruct", family: "Qwen", size: "0.5B" },
  { id: "Qwen/Qwen2.5-1.5B", name: "Qwen2.5 1.5B", family: "Qwen", size: "1.5B" },
  { id: "Qwen/Qwen2.5-1.5B-Instruct", name: "Qwen2.5 1.5B Instruct", family: "Qwen", size: "1.5B" },
  { id: "Qwen/Qwen2.5-3B", name: "Qwen2.5 3B", family: "Qwen", size: "3B" },
  { id: "Qwen/Qwen2.5-3B-Instruct", name: "Qwen2.5 3B Instruct", family: "Qwen", size: "3B" },
  { id: "Qwen/Qwen2.5-7B", name: "Qwen2.5 7B", family: "Qwen", size: "7B" },
  { id: "Qwen/Qwen2.5-7B-Instruct", name: "Qwen2.5 7B Instruct", family: "Qwen", size: "7B" },
  { id: "Qwen/Qwen2.5-14B", name: "Qwen2.5 14B", family: "Qwen", size: "14B" },
  { id: "Qwen/Qwen2.5-14B-Instruct", name: "Qwen2.5 14B Instruct", family: "Qwen", size: "14B" },
  { id: "Qwen/Qwen2.5-32B", name: "Qwen2.5 32B", family: "Qwen", size: "32B" },
  { id: "Qwen/Qwen2.5-32B-Instruct", name: "Qwen2.5 32B Instruct", family: "Qwen", size: "32B" },
  { id: "Qwen/Qwen2.5-72B", name: "Qwen2.5 72B", family: "Qwen", size: "72B" },
  { id: "Qwen/Qwen2.5-72B-Instruct", name: "Qwen2.5 72B Instruct", family: "Qwen", size: "72B" },
  { id: "Qwen/QwQ-32B-Preview", name: "QwQ 32B Preview", family: "Qwen", size: "32B" },

  // Phi Models
  { id: "microsoft/phi-1", name: "Phi-1", family: "Phi", size: "1.3B" },
  { id: "microsoft/phi-1_5", name: "Phi-1.5", family: "Phi", size: "1.3B" },
  { id: "microsoft/phi-2", name: "Phi-2", family: "Phi", size: "2.7B" },
  { id: "microsoft/Phi-3-mini-4k-instruct", name: "Phi-3 Mini 4K Instruct", family: "Phi", size: "3.8B" },
  { id: "microsoft/phi-4", name: "Phi-4", family: "Phi", size: "14B" },

  // Gemma Models
  { id: "google/gemma-2b", name: "Gemma 2B", family: "Gemma", size: "2B" },
  { id: "google/gemma-7b", name: "Gemma 7B", family: "Gemma", size: "7B" },
  { id: "google/gemma-2b-it", name: "Gemma 2B Instruct", family: "Gemma", size: "2B" },
  { id: "google/gemma-7b-it", name: "Gemma 7B Instruct", family: "Gemma", size: "7B" },
  { id: "google/gemma-2-2b", name: "Gemma 2 2B", family: "Gemma", size: "2B" },
  { id: "google/gemma-2-2b-it", name: "Gemma 2 2B Instruct", family: "Gemma", size: "2B" },
  { id: "google/gemma-2-9b", name: "Gemma 2 9B", family: "Gemma", size: "9B" },
  { id: "google/gemma-2-9b-it", name: "Gemma 2 9B Instruct", family: "Gemma", size: "9B" },
  { id: "google/gemma-2-27b", name: "Gemma 2 27B", family: "Gemma", size: "27B" },
  { id: "google/gemma-2-27b-it", name: "Gemma 2 27B Instruct", family: "Gemma", size: "27B" },

  // Bloom Models
  { id: "bigscience/bloom-560m", name: "Bloom 560M", family: "Bloom", size: "560M" },
  { id: "bigscience/bloom-1b1", name: "Bloom 1.1B", family: "Bloom", size: "1.1B" },
  { id: "bigscience/bloom-1b7", name: "Bloom 1.7B", family: "Bloom", size: "1.7B" },
  { id: "bigscience/bloom-3b", name: "Bloom 3B", family: "Bloom", size: "3B" },
  { id: "bigscience/bloom-7b1", name: "Bloom 7.1B", family: "Bloom", size: "7.1B" },

  // Yi Models
  { id: "01-ai/Yi-6B", name: "Yi 6B", family: "Yi", size: "6B" },
  { id: "01-ai/Yi-34B", name: "Yi 34B", family: "Yi", size: "34B" },
  { id: "01-ai/Yi-6B-Chat", name: "Yi 6B Chat", family: "Yi", size: "6B" },
  { id: "01-ai/Yi-34B-Chat", name: "Yi 34B Chat", family: "Yi", size: "34B" },

  // StableLM Models
  { id: "stabilityai/stablelm-base-alpha-3b", name: "StableLM Base Alpha 3B", family: "StableLM", size: "3B" },
  { id: "stabilityai/stablelm-base-alpha-7b", name: "StableLM Base Alpha 7B", family: "StableLM", size: "7B" },
  { id: "stabilityai/stablelm-tuned-alpha-3b", name: "StableLM Tuned Alpha 3B", family: "StableLM", size: "3B" },
  { id: "stabilityai/stablelm-tuned-alpha-7b", name: "StableLM Tuned Alpha 7B", family: "StableLM", size: "7B" },

  // TinyStories Models
  { id: "roneneldan/TinyStories-1M", name: "TinyStories 1M", family: "TinyStories", size: "1M" },
  { id: "roneneldan/TinyStories-3M", name: "TinyStories 3M", family: "TinyStories", size: "3M" },
  { id: "roneneldan/TinyStories-8M", name: "TinyStories 8M", family: "TinyStories", size: "8M" },
  { id: "roneneldan/TinyStories-28M", name: "TinyStories 28M", family: "TinyStories", size: "28M" },
  { id: "roneneldan/TinyStories-33M", name: "TinyStories 33M", family: "TinyStories", size: "33M" },
  { id: "roneneldan/TinyStories-Instruct-1M", name: "TinyStories Instruct 1M", family: "TinyStories", size: "1M" },
  { id: "roneneldan/TinyStories-Instruct-3M", name: "TinyStories Instruct 3M", family: "TinyStories", size: "3M" },
  { id: "roneneldan/TinyStories-Instruct-8M", name: "TinyStories Instruct 8M", family: "TinyStories", size: "8M" },
  { id: "roneneldan/TinyStories-Instruct-28M", name: "TinyStories Instruct 28M", family: "TinyStories", size: "28M" },
  { id: "roneneldan/TinyStories-Instruct-33M", name: "TinyStories Instruct 33M", family: "TinyStories", size: "33M" },

  // BERT Models
  { id: "google-bert/bert-base-cased", name: "BERT Base Cased", family: "BERT", size: "110M" },
  { id: "google-bert/bert-base-uncased", name: "BERT Base Uncased", family: "BERT", size: "110M" },
  { id: "google-bert/bert-large-cased", name: "BERT Large Cased", family: "BERT", size: "340M" },
  { id: "google-bert/bert-large-uncased", name: "BERT Large Uncased", family: "BERT", size: "340M" },

  // T5 Models
  { id: "google-t5/t5-small", name: "T5 Small", family: "T5", size: "60M" },
  { id: "google-t5/t5-base", name: "T5 Base", family: "T5", size: "220M" },
  { id: "google-t5/t5-large", name: "T5 Large", family: "T5", size: "770M" },

  // Other Models
  { id: "bigcode/santacoder", name: "SantaCoder", family: "Code", size: "1.1B" },
  { id: "ai-forever/mGPT", name: "mGPT", family: "mGPT", size: "1.3B" },
  { id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", name: "TinyLlama 1.1B Chat", family: "Llama", size: "1.1B" },
];

export const MODEL_FAMILIES = Array.from(new Set(TRANSFORMERLENS_MODELS.map(m => m.family))).sort();
