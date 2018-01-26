import matplotlib.pyplot as plt
import numpy as np
import pdb
import skimage.draw as draw
from scipy.misc import imresize, imsave

#imgs is [batch, y, x, f] and locs is [batch, numglimpse, 2] where last dim is
#y by x locs, with center as 0, 0, normalized to -1 to 1
def plotGlimpseTrace(imgs, locs, outdir, pix_ratio, targetSize=[180, 180], nameprefix=""):
    (nbatch, ny, nx, nf) = imgs.shape
    #TODO turn this off
    assert(ny == nx)
    #TODO make radius a function of image shape
    radius = 3
    (nbatch_2, nglimpse, nloc) = locs.shape
    assert(nbatch == nbatch)
    assert(nloc == 2)
    assert(nf == 3 or nf == 1)

    imgs = imgs.copy()
    locs = locs.copy()

    #Make grayscale into rgb
    if(nf == 1):
        imgs = np.tile(imgs, [1, 1, 1, 3])

    #Scale locs based on pix_ratio

    scale_loc = locs * 2 * pix_ratio

    #Convert locs to integer locs
    #Convert based on targetSize
    i_locs = (scale_loc + 1)/2
    y_locs = np.round(i_locs[..., 0] * targetSize[0]).astype(np.int32)
    x_locs = np.round(i_locs[..., 1] * targetSize[1]).astype(np.int32)

    for b in range(nbatch):
        img = imresize(imgs[b], targetSize).astype(np.float32)/255.0
        y_loc = y_locs[b]
        x_loc = x_locs[b]

        #Draw filled circle for first point
        rr, cc = draw.circle(y_loc[0], x_loc[0], radius, shape=targetSize)
        #Draw in green
        img[rr, cc, 1] = 1

        #Draw unfilled circle for last point
        rr, cc = draw.circle_perimeter(y_loc[-1], x_loc[-1], radius, shape=targetSize)
        img[rr, cc, 1] = 1

        #Draw lines following locations
        for i in range(nglimpse - 1):
            rr, cc, val = draw.line_aa(y_loc[i], x_loc[i], y_loc[i+1], x_loc[i+1])
            #Clip values
            rr = np.clip(rr, 0, targetSize[0]-1)
            cc = np.clip(cc, 0, targetSize[1]-1)

            img[rr, cc, 1] = val

        imsave(outdir + "/glimpse_" + nameprefix + "_" + str(b) + ".png", img)

