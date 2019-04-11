import tensorflow as tf 
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt 
import config
from model import encoder, generator, code_discriminator
import random
#from celebAIO import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data',one_hot=True);

'''
For better readability
I will be using same naming conventions as provided in
pseudo code of official paper.
'''

#Used to initialize kernel weights
stddev = 0.02;#99999;

tf.set_random_seed(0);
random.seed(0);

opts = config.config_mnist

enc_gen_learning_rate = opts['enc_gen_lr'];
code_discriminator_learning_rate = opts['code_disc_lr'];

batch_size = opts['batch_size'];

n_epoch = opts['num_epoch'];
z_dim = opts['z_dim'];

tfd = tf.contrib.distributions

#model_params
img_height = opts['img_height'];
img_width = opts['img_width'];
num_channels = opts['num_channels'];
n_inputs = 64*64; #as images are of 64 x 64 dimension
n_outputs = 10;

#Placeholder
#X = tf.placeholder(tf.float32,[None,img_height,img_width,num_channels]);
X = tf.placeholder(tf.float32,[None,784]);
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
l2_recons_loss = tf.reduce_mean(tf.pow(X - x_hat,2));

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

all_directories  = [log_directory, model_directory, output_directory];

for direc in all_directories:
    if not os.path.exists(direc):
        os.makedirs(direc);

def train():

    ###########################
    #DATA READING
    ###########################
    n_batches = mnist.train.num_examples/batch_size;
    n_batches = int(n_batches);
    #n_batches = 50;
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer());
        print('-'*80);
        print('n_batches : ',n_batches,' when batch_size : ',batch_size);
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
                    X_batch,_ = mnist.train.next_batch(batch_size);
                    fd = {X: X_batch};
                    _,code_disc_loss= sess.run([code_discriminator_train_optimizer,code_discriminator_loss],feed_dict = fd);

                #Train Encoder
                h=1;
                for i in range(h):
                    X_batch,_ = mnist.train.next_batch(batch_size);
                    fd = {X: X_batch};
                    _,_enc_gen_loss,merged= sess.run([enc_gen_train_optimizer,enc_gen_loss,merged_all],feed_dict = fd);

                if(iterations%20==0):
                    writer.add_summary(merged,iterations);

                if(batch%200 == 0):
                    print('Batch #',batch,' done!');

                #break;

            if(epoch%5==0):

                n = 5;
                
                reconstructed = np.empty((28*n,28*n));
                original = np.empty((28*n,28*n));
                generated = np.empty((28*n,28*n));
                
                
                for i in range(n):
                    
                    batch_X,_ = mnist.test.next_batch(n);
                    recons = sess.run(x_hat_test,feed_dict={X:batch_X});
                    #print ('recons : ',recons.shape);
                    recons = np.reshape(recons,[-1,784]);
                    #print ('recons : ',recons.shape);

                    sample = tf.random_normal([n,z_dim]);
                    generation = sess.run(x_hat_test,feed_dict={z_hat_test:sample.eval()});

                    for j in range(n):
                            original[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_X[j].reshape([28, 28]);

                    for j in range(n):
                        reconstructed[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons[j].reshape([28, 28]);

                    for j in range(n):
                            generated[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = generation[j].reshape([28, 28]);

                
                print("Generated Images");
                plt.figure(figsize=(n,n));
                plt.axis('off');
                plt.imshow(generated, origin="upper", cmap="gray");
                plt.title('Epoch '+str(epoch));
                plt.savefig('op/gen-img-'+str(epoch)+'.png');
                plt.close();

                print("Original Images");
                plt.figure(figsize=(n, n));
                plt.axis('off');
                plt.imshow(original, origin="upper", cmap="gray");
                plt.title('Epoch '+str(epoch));
                plt.savefig('op/orig-img-'+str(epoch)+'.png');
                plt.close();

                print("Reconstructed Images");
                plt.figure(figsize=(n, n));
                plt.axis('off');
                plt.imshow(reconstructed, origin="upper", cmap="gray");
                plt.title('Epoch '+str(epoch));
                plt.savefig('op/recons-img-'+str(epoch)+'.png');
                plt.close();

            if(epoch%5==0):

                save_path = saver.save(sess, model_directory+'/model_'+str(epoch));
                print("At epoch #",epoch," Model is saved at path: ",save_path);

            print('------------------------------------');
            print('=== Epoch #',epoch,' completed! ===');
            print('------------------------------------');


train();


