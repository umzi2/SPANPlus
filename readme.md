# SPANPlus
This repository is not an official modernization of [span](https://github.com/hongyuanyu/SPAN). The main idea of ​​which is to clean the code from unused pieces of code, increase stability by removing rgb_mean and adding a more stable upsampler. Also, the number of spab blocks can now be changed

Training code from [NeoSR](https://github.com/muslll/neosr)

| Name        | Upscaler     | blocks    | feature_channels |
|-------------|--------------|-----------|------------------|
| spanplus    | Dysample     | [4]       | 48               |
| spanplus-s  | DySample     | [2]       | 32               |
| spanplus-xl | DySample     | [4, 4, 4] | 96               |

### Detect:
```py 
def detect(state):
    state_keys = state.keys()
    n_feats = get_seq_len(state, "feats")-1
    blocks = [
        get_seq_len(state,f"feats.{n_feat+1}.block_n")
        for n_feat in range(n_feats)
    ]
    num_in_ch = state["conv_1.sk.weight"].shape[1]
    feature_channels = state["conv_1.eval_conv.weight"].shape[0]
    if "upsampler.0.weight" in state_keys:
        upsampler = "ps"
        num_out_ch = num_in_ch
        upscale = int((state["upsampler.0.weight"].shape[0] / num_in_ch) ** 0.5)
    else:
        upsampler = "lp"
        num_out_ch = state["upsampler.end_conv.weight"].shape[0]
        upscale = int((state["upsampler.offset.weight"].shape[0] // 8) ** 0.5)
    print(num_in_ch,
          num_out_ch,
          blocks,
          feature_channels,
          upscale,
          upsampler)
```
### TODO:
- release metrics and pretrain
