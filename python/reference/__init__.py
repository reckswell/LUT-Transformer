"""Library-built models used as controls. Nothing here is part of our model.

Files:
    torch_model.py  the core transformer built from torch.nn and autograd: the ground truth for
                    the from-scratch baseline (same weights in, same logits and gradients out)
    hf_gpt2.py      Hugging Face GPT-2, imported from `transformers` and never modified, plus the
                    export of its weights into our parameter layout (host.init_params names)
    hf_llama.py     the same for Hugging Face Llama (stories15M), with the rotary row-order
                    conversion between llama2.c and Hugging Face
    hf_profiles.py  which architectures our models can run (gpt2, llama), for eval/ and bench/
    local_cache.py  where downloads go: ../LUT-Transformer-local, never the repository or ~/.cache

baseline/ and LUT/ never import from here; tests/ and eval/ do. Keeping the library code out of
those folders keeps them purely from scratch (agent.md 2.1) and keeps each control independent of
the code it checks.
"""

from reference.local_cache import use_local_hf_cache

# Point Hugging Face's cache at ../LUT-Transformer-local before anything imports transformers or
# huggingface_hub (they read HF_HOME once, at import). See local_cache.py.
use_local_hf_cache()
