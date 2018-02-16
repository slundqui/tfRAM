Tensorflow implementation of Recurrent Models of Visual Attention (Mnih et al. 2014), with additional research. Code based off of https://github.com/zhongwen/RAM.

Reproduced results:

60 by 60 Translated MNIST
| Model                                          | Error  |
| ---------------------------------------------- | ------ |
| FC, 2 layers (64 hiddens each)                 | 6.78%  |
| FC, 2 layers (256 hiddens each)                | 2.65%  |
| Convolutional, 2 layers                        | 1.57%  |
| RAM, 4 glimpses, $12 \times 12$, 3 scale       | 1.54%  |
| RAM, 6 glimpses, $12 \times 12$, 3 scale       | 1.08%  |
| RAM, 8 glimpses, $12 \times 12$, 3 scale       | 0.94%  |

| 60 by 60 Cluttered Translated MNIST
| Model                                          | Error  |
| ---------------------------------------------- | ------ |
| FC, 2 layers (64 hiddens each)                 | 29.13% |
| FC, 2 layers (256 hiddens each)                | 11.36% |
| Convolutional, 2 layers                        | 8.37%  |
| RAM, 4 glimpses, $12 \times 12$, 3 scale       | 5.15%  |
| RAM, 6 glimpses, $12 \times 12$, 3 scale       | 3.33%  |
| RAM, 8 glimpses, $12 \times 12$, 3 scale       | 2.63%  |

100 by 100$ Cluttered Translated MNIST
| Model                                          | Error  |
| ---------------------------------------------- | ------ |
| Convolutional, 2 layers                        | 16.22% |
| RAM, 4 glimpses, $12 \times 12$, 3 scale       | 14.86% |
| RAM, 6 glimpses, $12 \times 12$, 3 scale       | 8.3%   |
| RAM, 8 glimpses, $12 \times 12$, 3 scale       | 5.9%   |

60 by 60 Cluttered MNIST 6 glimpses examples
| Mean output                                                               | Sampled output |
| ----------------------------------------------                            | -------------- |
|![mean0](https://github.com/slundqui/tfRAM/readme_imgs/glimpse_mean_0.png) | ![samp0](https://github.com/slundqui/tfRAM/readme_imgs/glimpse_sampled_0.png) |
|![mean0](https://github.com/slundqui/tfRAM/readme_imgs/glimpse_mean_1.png) | ![samp0](https://github.com/slundqui/tfRAM/readme_imgs/glimpse_sampled_1.png) |
|![mean0](https://github.com/slundqui/tfRAM/readme_imgs/glimpse_mean_2.png) | ![samp0](https://github.com/slundqui/tfRAM/readme_imgs/glimpse_sampled_2.png) |
|![mean0](https://github.com/slundqui/tfRAM/readme_imgs/glimpse_mean_3.png) | ![samp0](https://github.com/slundqui/tfRAM/readme_imgs/glimpse_sampled_3.png) |
|![mean0](https://github.com/slundqui/tfRAM/readme_imgs/glimpse_mean_4.png) | ![samp0](https://github.com/slundqui/tfRAM/readme_imgs/glimpse_sampled_4.png) |

