Tensorflow implementation of Recurrent Models of Visual Attention (Mnih et al. 2014), with additional research. Code based off of https://github.com/zhongwen/RAM.

#Reproduced results

##60 by 60 Translated MNIST
<table>
  <tr>
    <th>Model</th>
    <th>Error</th>
  </tr>
  <tr>
    <td> FC, 2 layers (64 hiddens each)           </td>      <td> 6.78%  </td>
  </tr>
  <tr>
    <td> FC, 2 layers (256 hiddens each)          </td>      <td> 2.65%  </td>
  </tr>
  <tr>
    <td> Convolutional, 2 layers                  </td>      <td> 1.57%  </td>
  </tr>
  <tr>
    <td> RAM, 4 glimpses, $12 \times 12$, 3 scale </td>      <td> 1.54%  </td>
  </tr>
  <tr>
    <td> RAM, 6 glimpses, $12 \times 12$, 3 scale </td>      <td> 1.08%  </td>
  </tr>
  <tr>
    <td> RAM, 8 glimpses, $12 \times 12$, 3 scale </td>      <td> 0.94%  </td>
  </tr>
</table>

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
|![mean0](https://github.com/slundqui/tfRAM/tree/master/readme_imgs/glimpse_mean_0.png)| ![samp0](https://github.com/slundqui/tfRAM/tree/master/readme_imgs/glimpse_sampled_0.png) |
|![mean1](https://github.com/slundqui/tfRAM/tree/master/readme_imgs/glimpse_mean_0.png)| ![samp1](https://github.com/slundqui/tfRAM/tree/master/readme_imgs/glimpse_sampled_0.png) |
|![mean2](https://github.com/slundqui/tfRAM/tree/master/readme_imgs/glimpse_mean_0.png)| ![samp2](https://github.com/slundqui/tfRAM/tree/master/readme_imgs/glimpse_sampled_0.png) |
|![mean3](https://github.com/slundqui/tfRAM/tree/master/readme_imgs/glimpse_mean_0.png)| ![samp3](https://github.com/slundqui/tfRAM/tree/master/readme_imgs/glimpse_sampled_0.png) |
|![mean4](https://github.com/slundqui/tfRAM/tree/master/readme_imgs/glimpse_mean_0.png)| ![samp4](https://github.com/slundqui/tfRAM/tree/master/readme_imgs/glimpse_sampled_0.png) |

