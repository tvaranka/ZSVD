## Main `inference_path.py`
`L39` Encodees the raw frames to latent frames so that they can be inverted to latent noise

`L44` Ensures that for inversion the default block and attention processor are used, i.e., there is no attention switching.

`L55` Performs inversion, where `zs` contains the DDPM noises $\epsilon_t$ and `wts` contains the latents noises $x_t$.

`L66` Can be used to modify the blocks used in attention switching.

`L71` `apply_mod_block` inserts the attention processors `CogVideoXAttnProcessor2_0Mod` that perform the attention switching to the correct blocks.

`L81` Performs deraining starting from the `skip` value

## Modifying the attention process `utils.py`

#### `CogVideoXAttnProcessor2_0Mod`
`L266` Create cross-conditional the cross-conditional features (eq. 6 from paper) to ensure that the text and image features are aligned before switching.
`L310` Switch keys and values for the text features from cross-conditional feature to the derained features.
```python
# Batch 0 is derained feature and batch 2 is cross-conditional feature
key[0, :text_seq_length] = key[2, :text_seq_length]
value[0, :text_seq_length] = value[2, :text_seq_length]
```
`L313` Remove cross-conditional feature and continue with attention.

#### `CogVideoXBlockMod`
As the text features are swapped to null in early blocks, but are still required in the middle blocks, the text features need to be saved in the earlier blocks. 

`CogVideoXBlockMod` splits the computation of attention to two by first computing the text features and then the image features.
In the first attention (L446) for text features standard attention is used, but for the image features (L453) the modified attention from above is used (if the modified attention was set for the block).

This split enables the block to retain the negative text features even if they were set to null in the latter attention, while still being able to use the null text features to modify the image features.

