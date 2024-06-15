# SPANPlus
This repository is not an official modernization of [span](https://github.com/hongyuanyu/SPAN). The main idea of ​​which is to clean the code from unused pieces of code, increase stability by removing rgb_mean and adding a more stable upsampler. Also, the number of spab blocks can now be changed

| Name        | Upscaler     | n_blocks |
|-------------|--------------|----------|
| spanplus    | Dysample     | 4        |
| spanplus-s  | DySample     | 2        | 
| spanplus-st | PixelShuffle | 4        |


P.S. spanplus-st - essentially a regular span, the output will be identical, but it does not have reverse support due to different block names
