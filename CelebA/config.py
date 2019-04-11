config_celebA = {};

config_celebA['enc_gen_lr'] = 0.0003;
config_celebA['code_disc_lr'] = 0.001;
config_celebA['batch_size'] = 128;
config_celebA['z_dim'] = 64;
config_celebA['lamda'] = 1;
config_celebA['img_height'] = 64;
config_celebA['img_width'] = 64;
config_celebA['num_channels'] = 3;
config_celebA['crop_style'] = 'closecrop'#resizecrop for AGE

config_celebA['dataset_name'] = 'celebA';
config_celebA['dataset_split_name'] = 'train';
config_celebA['dataset_dir'] = '/home/test/Desktop/pm/img_align_celeba/';#'./f1/'
config_celebA['shuffle_data'] = True; # accepts True or False
config_celebA['num_epoch'] = 200;
config_celebA['celebA_crop'] = 'closecrop'#'closecrop'
config_celebA['train_min_filenum'] = 1;
config_celebA['train_max_filenum'] = 162770;
config_celebA['val_min_filenum'] = 162771;
config_celebA['val_max_filenum'] = 182637;
config_celebA['load_model_number'] = 60;



