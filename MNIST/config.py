config_mnist = {};

config_mnist['enc_gen_lr'] = 0.001;
config_mnist['code_disc_lr'] = 0.0005;
config_mnist['batch_size'] = 128;
config_mnist['z_dim'] = 8;
config_mnist['lamda'] = 1;
config_mnist['img_height'] = 28;
config_mnist['img_width'] = 28;
config_mnist['num_channels'] = 1;
config_mnist['crop_style'] = 'closecrop'#resizecrop for AGE

config_mnist['dataset_name'] = 'celebA';
config_mnist['dataset_split_name'] = 'train';
config_mnist['dataset_dir'] = '/home/test/Desktop/pm/img_align_celeba/';#'./f1/'
config_mnist['shuffle_data'] = True; # accepts True or False
config_mnist['num_epoch'] = 100;
config_mnist['celebA_crop'] = 'closecrop'#'closecrop'
config_mnist['train_min_filenum'] = 1;
config_mnist['train_max_filenum'] = 162770;
config_mnist['val_min_filenum'] = 162771;
config_mnist['val_max_filenum'] = 182637;
config_mnist['load_model_number'] = 60;



