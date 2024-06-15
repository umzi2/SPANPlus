# SPANPlus
This repository is not an official modernization of span. The main idea of ​​which is to clean the code from unused pieces of code, increase stability by removing rgb_mean and adding a more stable upsampler. Also, the number of spab blocks can now be changed

| Name        | Upscaler     | n_blocks |
|-------------|--------------|----------|
| spanplus    | Dysample     | 4        |
| spanplus-s  | DySample     | 2        | 
| spanplus-st | PixelShuffle | 4        |
P.S. spanplus-st - essentially a regular span, the output will be identical, but it does not have reverse support due to different block names

### Detect:
```py 
def detect(state):
    state_keys = state.keys()
    n_blocks = get_seq_len(state, "block_n")
    num_in_ch = state["conv_1.sk.weight"].shape[1]
    feature_channels = state["block_1.c1_r.eval_conv.weight"].shape[0]
    if "upsampler.0.weight" in state_keys:
        upsampler = "ps"
        num_out_ch = num_in_ch
        upscale = int((state["upsampler.0.weight"].shape[0] / num_in_ch) ** 0.5)
    else:
        upsampler = "lp"
        num_out_ch = state["upsampler.end_conv.1.weight"].shape[0]
        upscale = int((state["upsampler.offset.weight"].shape[0] // 8) ** 0.5)
    print(num_in_ch,
          num_out_ch,
          n_blocks,
          feature_channels,
          upscale,
          upsampler)
```
### TODO:
- release metrics and pretrain