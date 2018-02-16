Tensorflow implementation of Recurrent Models of Visual Attention (Mnih et al. 2014), with additional research. Code based off of https://github.com/zhongwen/RAM.

<h2>Results</h2>

<h3>60 by 60 Translated MNIST</h3>
<table>
  <tr><th> Model                                    </th><th> Error  </th></tr>
  <tr><td> FC, 2 layers (64 hiddens each)           </td><td> 6.78%  </td></tr>
  <tr><td> FC, 2 layers (256 hiddens each)          </td><td> 2.65%  </td></tr>
  <tr><td> Convolutional, 2 layers                  </td><td> 1.57%  </td></tr>
  <tr><td> RAM, 4 glimpses, $12 \times 12$, 3 scale </td><td> 1.54%  </td></tr>
  <tr><td> RAM, 6 glimpses, $12 \times 12$, 3 scale </td><td> 1.08%  </td></tr>
  <tr><td> RAM, 8 glimpses, $12 \times 12$, 3 scale </td><td> 0.94%  </td></tr>
</table>

<h3> 60 by 60 Cluttered Translated MNIST </h3>
<table>
  <tr><th> Model                                     </th><th> Error  </th></tr>
  <tr><td> FC, 2 layers (64 hiddens each)            </td><td> 29.13% </td></tr>
  <tr><td> FC, 2 layers (256 hiddens each)           </td><td> 11.36% </td></tr>
  <tr><td> Convolutional, 2 layers                   </td><td> 8.37%  </td></tr>
  <tr><td> RAM, 4 glimpses, $12 \times 12$, 3 scale  </td><td> 5.15%  </td></tr>
  <tr><td> RAM, 6 glimpses, $12 \times 12$, 3 scale  </td><td> 3.33%  </td></tr>
  <tr><td> RAM, 8 glimpses, $12 \times 12$, 3 scale  </td><td> 2.63%  </td></tr>
</table>

<h3> 100 by 100$ Cluttered Translated MNIST </h3>
<table>
  <tr><th> Model                                     </th><th> Error  </th></tr>
  <tr><td> Convolutional, 2 layers                   </td><td> 16.22% </td></tr>
  <tr><td> RAM, 4 glimpses, $12 \times 12$, 3 scale  </td><td> 14.86% </td></tr>
  <tr><td> RAM, 6 glimpses, $12 \times 12$, 3 scale  </td><td> 8.3%   </td></tr>
  <tr><td> RAM, 8 glimpses, $12 \times 12$, 3 scale  </td><td> 5.9%   </td></tr>
</table>

<h3>60 by 60 Cluttered MNIST 6 glimpses examples </h3>
Solid square is first glimpse, line is path of attention, circle is last glimpse.
<table>
  <tr><th> Mean output                                           </th><th> Sampled output                              </th></tr>
  <tr><td><img src="readme_imgs/glimpse_mean_0.png" alt="mean0"> </td><td> <img src="readme_imgs/glimpse_sampled_0.png" alt="samp0"> </td></tr>
  <tr><td><img src="readme_imgs/glimpse_mean_1.png" alt="mean1"> </td><td> <img src="readme_imgs/glimpse_sampled_1.png" alt="samp1"> </td></tr>
  <tr><td><img src="readme_imgs/glimpse_mean_2.png" alt="mean2"> </td><td> <img src="readme_imgs/glimpse_sampled_2.png" alt="samp2"> </td></tr>
  <tr><td><img src="readme_imgs/glimpse_mean_3.png" alt="mean3"> </td><td> <img src="readme_imgs/glimpse_sampled_3.png" alt="samp3"> </td></tr>
  <tr><td><img src="readme_imgs/glimpse_mean_4.png" alt="mean4"> </td><td> <img src="readme_imgs/glimpse_sampled_4.png" alt="samp4"> </td></tr>

