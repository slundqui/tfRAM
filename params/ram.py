#Params object for mono_ram

class RamParams(object):
    ##Bookkeeping params
    #Base output directory
    out_dir            = "/home/slundquist/mountData/ram/"
    #Inner run directory
    run_dir            = out_dir + "/mono_ram/"

    #Save parameters
    save_period        = 100000
    #output plots directory
    plot_period        = 100000
    eval_period        = 1718 # 1 epoch
    #Progress step
    progress           = 100
    #Controls how often to write out to tensorboard
    write_step         = 300
    #Flag for loading weights from checkpoint
    load               = False
    load_file          = ""
    #Device to run on
    device             = None #device

    #data params
    num_train_examples = None #dataObj.num_train_examples

    #RAM params
    win_size           = 8       #The size of each glimpse in pixels in both x and y dimension
    batch_size         = 32      #Batch size of training
    eval_batch_size    = 50      #Batch size of testing
    loc_std            = 0.05     #Standard deviation of random noise added to locations
    det_eval           = False   #If true, use only mean of location network for evaluation
    original_size      = None    #dataObj.inputShape #Size of the input image in (y, x, f)
    glimpse_scales     = 1       #Number of channels in input image
    sensor_size        = win_size**2 * glimpse_scales #Total size of input glimpse
    hg_size            = 128     #Number of features in first layer for glimpse encode
    hl_size            = 128     #Number of features in first layer for location encode
    g_size             = 256     #Number of features in second layer (combine g and l)
    loc_dim            = 2       #Number of dimensions used in the location output
    cell_size          = 256     #Size of hidden latent space in LSTM
    num_glimpses       = 6       #Number of total glimpses
    num_classes        = 10      #Number of output classes
    max_grad_norm      = 5.      #Clipping norm for gradient clipping

    num_steps          = 500001  #Number of total steps
    lr_start           = 1e-3    #Starting learning rate for lr decay
    lr_min             = 1e-4    #Minimum learning rate for lr decay
    lr_decay           = .99     #Learning rate decay multiplier

    # Monte Carlo sampling
    M                  = 10
