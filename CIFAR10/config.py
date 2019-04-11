config_cifar10 = {};

config_cifar10['dataset_dir'] = './cifar-10-batches-py/';
config_cifar10['dataset_name'] = 'CIFAR10';
config_cifar10['enc_gen_lr'] = 0.0002;
config_cifar10['code_disc_lr'] = 0.0005;
config_cifar10['batch_size'] = 128;
config_cifar10['n_epoch'] = 100;
config_cifar10['z_dim'] = 100;
config_cifar10['lamda'] = 1;
config_cifar10['img_height'] = 32;
config_cifar10['img_width'] = 32;
config_cifar10['num_channels'] = 3;
config_cifar10['crop_style'] = 'closecrop'#resizecrop for AGE


