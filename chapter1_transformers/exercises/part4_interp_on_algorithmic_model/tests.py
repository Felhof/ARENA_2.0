# %%
from typing import Callable
import torch as t
from torch import Tensor
from jaxtyping import Float, Int
from transformer_lens import ActivationCache, HookedTransformer
from part4_interp_on_algorithmic_model.brackets_datasets import BracketsDataset

# from IPython import get_ipython
# ipython = get_ipython()
# # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
# ipython.magic("load_ext autoreload")
# ipython.magic("autoreload 2")

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")

t.set_grad_enabled(False)

# %%

def test_get_activations(get_activations: Callable, model: HookedTransformer, data: BracketsDataset):

    import part4_interp_on_algorithmic_model.solutions as solutions
    
    embed_actual = get_activations(model, data.toks, "hook_embed")
    embed_expected = solutions.get_activations(model, data.toks, "hook_embed")
    t.testing.assert_close(embed_actual, embed_expected)
    
    dict_actual = get_activations(model, data.toks, ["hook_embed", "hook_pos_embed"])
    dict_expected = solutions.get_activations(model, data.toks, ["hook_embed", "hook_pos_embed"])
    assert isinstance(dict_actual, ActivationCache)
    t.testing.assert_close(dict_actual["hook_pos_embed"], dict_expected["hook_pos_embed"])
    
    print("All tests in `test_get_activations` passed!")

def test_get_ln_fit(get_ln_fit: Callable, model: HookedTransformer, data: BracketsDataset):

    import part4_interp_on_algorithmic_model.solutions as solutions

    for seq_pos in [1, 0, None]:
        fit_actual = get_ln_fit(model, data, model.ln_final, seq_pos)
        fit_expected = solutions.get_ln_fit(model, data, model.ln_final, seq_pos)
        t.testing.assert_close(fit_actual[0].coef_, fit_expected[0].coef_)
    
    print("All tests in `test_get_ln_fit` passed!")

# %%

def test_get_pre_final_ln_dir(get_pre_final_ln_dir: Callable, model: HookedTransformer, data: BracketsDataset):
    import part4_interp_on_algorithmic_model.solutions as solutions
    dir_actual = get_pre_final_ln_dir(model, data)
    dir_expected = solutions.get_pre_final_ln_dir(model, data)
    t.testing.assert_close(dir_actual, dir_expected)
    print("All tests in `test_get_pre_final_ln_dir` passed!")

# %%

def test_get_out_by_components(get_out_by_components: Callable, model: HookedTransformer, data: BracketsDataset):
    import part4_interp_on_algorithmic_model.solutions as solutions
    dir_actual = get_out_by_components(model, data)
    dir_expected = solutions.get_out_by_components(model, data)
    t.testing.assert_close(dir_actual.sum(1), dir_expected.sum(1))
    print("All tests in `test_get_out_by_components` passed!")

# %%

def test_out_by_component_in_unbalanced_dir(out_by_component_in_unbalanced_dir: Float[Tensor, "comp batch"], model: HookedTransformer, data: BracketsDataset):

    import part4_interp_on_algorithmic_model.solutions as solutions

    out_by_components_seq0: Float[Tensor, "comp batch d_model"] = solutions.get_out_by_components(model, data)[:, :, 0, :]

    pre_final_ln_dir: Float[Tensor, "d_model"] = solutions.get_pre_final_ln_dir(model, data)

    out_by_component_in_unbalanced_dir_expected: Float[Tensor, "comp batch"] = out_by_components_seq0 @ pre_final_ln_dir

    out_by_component_in_unbalanced_dir_expected -= out_by_component_in_unbalanced_dir_expected[:, data.isbal].mean(dim=1).unsqueeze(1)

    t.testing.assert_close(out_by_component_in_unbalanced_dir, out_by_component_in_unbalanced_dir_expected)

    print("All tests in `test_out_by_component_in_unbalanced_dir` passed!")

# %%

def test_get_post_final_ln_dir(get_post_final_ln_dir, model: HookedTransformer):
    import part4_interp_on_algorithmic_model.solutions as solutions
    actual = get_post_final_ln_dir(model)
    expected = solutions.get_post_final_ln_dir(model)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_get_post_final_ln_dir` passed!")

def test_total_elevation_and_negative_failures(data: BracketsDataset, total_elevation_failure: Float[Tensor, "batch"], negative_failure: Float[Tensor, "batch"]):

    import part4_interp_on_algorithmic_model.solutions as solutions

    tef_expected, nf_expected = solutions.is_balanced_vectorized_return_both(data.toks)

    t.testing.assert_close(negative_failure, nf_expected)
    t.testing.assert_close(total_elevation_failure, tef_expected, msg="total_elevation_failure is not correct. Did you remember to read the sequence from right to left?")

    print("All tests in `test_total_elevation_and_negative_failures` passed!")

# %%

def test_get_attn_probs(get_attn_probs: Callable, model: HookedTransformer, data: BracketsDataset):
    import part4_interp_on_algorithmic_model.solutions as solutions
    probs_actual = get_attn_probs(model, data, 2, 0)
    probs_expected = solutions.get_attn_probs(model, data, 2, 0)
    t.testing.assert_close(probs_actual, probs_expected)
    print("All tests in `test_get_attn_probs` passed!")

# %%

def test_get_WOV(get_WOV: Callable, model: HookedTransformer):

    W_OV_00 = get_WOV(model, 0, 0)
    W_OV_00_expected = model.W_V[0, 0] @ model.W_O[0, 0]

    t.testing.assert_close(W_OV_00, W_OV_00_expected)
    print("All tests in `test_get_WOV` passed!")

def test_get_pre_20_dir(get_pre_20_dir: Callable, model: HookedTransformer, data: BracketsDataset):
    import part4_interp_on_algorithmic_model.solutions as solutions
    dir_actual = get_pre_20_dir(model, data)
    dir_expected = solutions.get_pre_20_dir(model, data)
    t.testing.assert_close(dir_actual, dir_expected)
    print("All tests in `test_get_pre_20_dir` passed!")

def test_get_out_by_neuron(get_out_by_neuron: Callable, model: HookedTransformer, data: BracketsDataset):
    import part4_interp_on_algorithmic_model.solutions as solutions
    out = get_out_by_neuron(model, data, layer=0, seq=1)
    out_expected = solutions.get_out_by_neuron(model, data, layer=0, seq=1)
    t.testing.assert_close(out, out_expected)
    print("All tests in `test_get_out_by_neuron` passed!")

def test_get_out_by_neuron_in_20_dir(get_out_by_neuron_in_20_dir: Callable, model: HookedTransformer, data: BracketsDataset):
    import part4_interp_on_algorithmic_model.solutions as solutions
    out = get_out_by_neuron_in_20_dir(model, data, layer=0)
    out_expected = solutions.get_out_by_neuron_in_20_dir(model, data, layer=0)
    t.testing.assert_close(out, out_expected)
    print("All tests in `test_get_out_by_neuron_in_20_dir` passed!")

def test_get_out_by_neuron_in_20_dir_less_memory(get_out_by_neuron_in_20_dir_less_memory: Callable, model: HookedTransformer, data: BracketsDataset):
    import part4_interp_on_algorithmic_model.solutions as solutions
    out = get_out_by_neuron_in_20_dir_less_memory(model, data, layer=0)
    out_expected = solutions.get_out_by_neuron_in_20_dir_less_memory(model, data, layer=0)
    t.testing.assert_close(out, out_expected)
    print("All tests in `test_get_out_by_neuron_in_20_dir_less_memory` passed!")

def test_get_q_and_k_for_given_input(get_q_and_k_for_given_input, model, tokenizer):
    import part4_interp_on_algorithmic_model.solutions as solutions
    parens = "()"
    q, k = get_q_and_k_for_given_input(model, tokenizer, parens, 0, 0)
    q_expected, k_expected = solutions.get_q_and_k_for_given_input(model, tokenizer, parens, 0, 0)
    t.testing.assert_close(q, q_expected)
    t.testing.assert_close(k, k_expected)
    print("All tests in `test_get_q_and_k_for_given_input` passed!")