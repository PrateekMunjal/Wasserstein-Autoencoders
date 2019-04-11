import tensorflow as tf 
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt 
import config
from model import encoder, generator, code_discriminator
import random
from cifarIO import *

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('tmp/data/',one_hot=True);

tf.set_random_seed(0);
random.seed(0);
'''
For better readability
I will be using same naming conventions as provided in
pseudo code of official paper.
'''

#Used to initialize kernel weights
stddev = 0.02;#99999;

opts = config.config_cifar10

enc_gen_learning_rate = opts['enc_gen_lr'];
code_discriminator_learning_rate = opts['code_disc_lr'];

batch_size = opts['batch_size'];

n_epoch = opts['n_epoch'];
z_dim = opts['z_dim'];

tfd = tf.contrib.distributions

#model_params
n_inputs = 28*28; #as images are of 28 x 28 dimension
img_height = opts['img_height'];
img_width = opts['img_width'];
num_channels = opts['num_channels'];
n_outputs = 10;

#Placeholder
X = tf.placeholder(tf.float32,[None,img_height,img_width,num_channels]);
Y = tf.placeholder(tf.float32,[None,n_outputs]);

def prior_z(latent_dim):
        z_mean = tf.zeros(latent_dim);
        z_var = tf.ones(latent_dim);
        return tfd.MultivariateNormalDiag(z_mean,z_var);

prior_dist = prior_z(z_dim);

z_prime = prior_dist.sample(batch_size);

# z_hat ~ Q(Z|X) variational distribution parametrized by encoder network
z_hat = encoder(X);
z_hat_test = encoder(X,isTrainable=False,reuse=True);
x_hat = generator(z_hat);
x_hat_test = generator(z_hat_test,isTrainable=False,reuse=True);
x_prime = generator(z_prime,reuse=True);

z_hat_logits = code_discriminator(z_hat);
z_prime_logits = code_discriminator(z_prime,reuse=True);

lamda = opts['lamda'];
l2_recons_loss = 0.05*tf.reduce_mean(tf.pow(X - x_hat,2));

# encoder_loss = lamda_enc*l1_recons_loss + RC_w(z_hat_logits);

# generator_loss = lamda_gen*l1_recons_loss + RD_phi(x_hat_logits) + RD_phi(x_prime_logits);

enc_adversary_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z_hat_logits, labels=tf.ones_like(z_hat_logits)));

code_disc_real_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z_prime_logits, labels=tf.ones_like(z_prime_logits)));
code_disc_fake_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z_hat_logits, labels=tf.zeros_like(z_hat_logits)));

code_discriminator_loss = lamda*(code_disc_real_term + code_disc_fake_term);

enc_gen_loss = l2_recons_loss + lamda*enc_adversary_loss;

eta_encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='phi_encoder');
theta_generator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='theta_generator');
enc_gen_params = eta_encoder_params + theta_generator_params;
gamma_code_discriminator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='gamma_code_discriminator');

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);
with tf.control_dependencies(update_ops):
    enc_gen_optimizer = tf.train.AdamOptimizer(learning_rate=enc_gen_learning_rate,beta1=0.5);
    enc_gen_gradsVars = enc_gen_optimizer.compute_gradients(enc_gen_loss, enc_gen_params);
    enc_gen_train_optimizer = enc_gen_optimizer.apply_gradients(enc_gen_gradsVars);

    code_discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=code_discriminator_learning_rate,beta1=0.5);
    code_discriminator_gradsVars = code_discriminator_optimizer.compute_gradients(code_discriminator_loss, gamma_code_discriminator_params);
    code_discriminator_train_optimizer = code_discriminator_optimizer.apply_gradients(code_discriminator_gradsVars);

tf.summary.scalar("enc_gen_loss",enc_gen_loss);
tf.summary.scalar("code_discriminator_loss",code_discriminator_loss);
tf.summary.scalar("l2_recons_loss",l2_recons_loss);
all_gradsVars = [enc_gen_gradsVars, code_discriminator_gradsVars];

for grad_vars in all_gradsVars:
    for g,v in grad_vars:  
        tf.summary.histogram(v.name,v)
        tf.summary.histogram(v.name+str('grad'),g)

merged_all = tf.summary.merge_all();
log_directory = './myWAE-GAN-dir';
model_directory='./myWAE-GAN-model_dir';
output_directory = './op/';
#train_output_directory = './train-op/';

all_directories  = [log_directory, model_directory, output_directory];

for direc in all_directories:
    if not os.path.exists(direc):
        os.makedirs(direc);

def train():

    # X_train = load_data();
    # X_test = load_test_data();
    X_train = load_data();
    X_train = normalize_image(X_train);

    X_test = load_test_data();
    X_test = normalize_image(X_test);
    
    n_batches = X_train.shape[0]/batch_size;#mnist.train.num_examples/batch_size;
    n_batches = int(n_batches);

    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer());
        print('-'*80);
        print('n_batches : ',n_batches,' when batch_size: ',batch_size);
        #for tensorboard
        saver = tf.train.Saver(max_to_keep=3);
        writer = tf.summary.FileWriter(log_directory,sess.graph);
        iterations = 0;
        
        for epoch in range(n_epoch):

            for batch in range(n_batches):
                iterations += 1;
                
                #Train Code Discriminator
                l=1;
                for i in range(l):
                    X_batch = get_cifar_batch(X_train,batch_size);
                    fd = {X: X_batch};
                    _,code_disc_loss= sess.run([code_discriminator_train_optimizer,code_discriminator_loss],feed_dict = fd);

                #Train Encoder
                h=1;
                for i in range(h):
                    X_batch = get_cifar_batch(X_train,batch_size);
                    fd = {X: X_batch};
                    _,_enc_gen_loss,merged= sess.run([enc_gen_train_optimizer,enc_gen_loss,merged_all],feed_dict = fd);

                if(iterations%20==0):
                    writer.add_summary(merged,iterations);

                if(batch%200 == 0):
                    print('Batch #',batch,' done!');

                #break;

            if(epoch%2==0):

                num_val_img = 25;
                batch_X = get_cifar_batch(X_test,batch_size);
                
                recons = sess.run(x_hat_test,feed_dict={X:batch_X});
                recons = np.reshape(recons,[-1,32,32,3]);



                n_gen = 25;
                sample = tf.random_normal([n_gen,z_dim]);
                generations = sess.run(x_hat_test,feed_dict={z_hat_test:sample.eval()});
                generations = np.reshape(generations,[-1,32,32,3]);

                temp_index = -1;
                for s in range(generations.shape[0]):
                    temp_index += 1;
                    generations[temp_index] = denormalize_image(generations[temp_index]);

                temp_index = -1;
                for s in range(batch_X.shape[0]):
                    temp_index += 1;
                    batch_X[temp_index] = denormalize_image(batch_X[temp_index]);

                temp_index = -1;
                for s in range(recons.shape[0]):
                    temp_index += 1;
                    recons[temp_index] = denormalize_image(recons[temp_index]);

                n = 5;
                reconstructed = np.empty((32*n,32*n,3));
                original = np.empty((32*n,32*n,3));
                
                for i in range(n):
                    for j in range(n):
                        original[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = batch_X[i*n+j];#.reshape([32, 32,3]);
                        reconstructed[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = recons[i*n+j];
                        #generated_images[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = generations[i*n+j];

                n1 = 5;
                generated_images = np.empty((32*n1,32*n1,3));
                for i in range(n1):
                    for j in range(n1):
                        generated_images[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = generations[i*n1+j];

                print("Original Images");
                plt.figure(figsize=(n, n));
                plt.imshow(original, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig(output_directory+'orig-img-'+str(epoch)+'.png');
                plt.close();

                print("Reconstructed Images");
                plt.figure(figsize=(n, n));
                plt.imshow(reconstructed, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig(output_directory+'recons-img-'+str(epoch)+'.png');
                plt.close();

                print("Generated Images");
                plt.figure(figsize=(n1, n1));
                plt.title("Epoch "+str(epoch));
                plt.imshow(generated_images, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig(output_directory+'gen-img-'+str(epoch)+'.png');
                plt.close();

            if(epoch%5==0):

                save_path = saver.save(sess, model_directory+'/model_'+str(epoch));
                print("At epoch #",epoch," Model is saved at path: ",save_path);

            print('------------------------------------');
            print('=== Epoch #',epoch,' completed! ===');
            print('------------------------------------');


train();


