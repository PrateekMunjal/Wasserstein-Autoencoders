import tensorflow as tf 
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt 
import config

stddev = 0.00999;
opts = config.config_mnist

z_dim = opts['z_dim'];
img_height = opts['img_height'];
img_width = opts['img_width'];
num_channels = opts['num_channels'];

def encoder(X,isTrainable=True,reuse=False,name='phi_encoder'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables();

        X = tf.reshape(X,[-1,28,28,1]);

        conv1 = tf.layers.conv2d(X,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=16,kernel_size=[3,3],padding='SAME',strides=(2,2),name='enc_conv1_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv1 = tf.layers.batch_normalization(conv1,training=isTrainable,reuse=reuse,name='bn_1');
        conv1 = tf.nn.relu(conv1,name='leaky_relu_conv_1');

        #14x14x32
        conv2 = tf.layers.conv2d(conv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=32,kernel_size=[3,3],padding='SAME',strides=(2,2),name='enc_conv2_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv2 = tf.layers.batch_normalization(conv2,training=isTrainable,reuse=reuse,name='bn_2');
        conv2 = tf.nn.relu(conv2,name='leaky_relu_conv_2');
        
        #7x7x64
        conv3 = tf.layers.conv2d(conv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[3,3],padding='SAME',strides=(2,2),name='enc_conv3_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv3 = tf.layers.batch_normalization(conv3,training=isTrainable,reuse=reuse,name='bn_3');
        conv3 = tf.nn.relu(conv3,name='leaky_relu_conv_3');
    
        
        #4x4x128
        conv3_flattened = tf.layers.flatten(conv3);
        
        latent_code = tf.layers.dense(conv3_flattened,z_dim,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),activation=None,name='enc_latent_space',trainable=isTrainable,reuse=reuse);
        return latent_code; 

def generator(z_sample,isTrainable=True,reuse=False,name='theta_generator'):
    with tf.variable_scope(name) as scope:  
        #decoder_activations = {};
        if reuse:
            scope.reuse_variables();

        z_sample = tf.layers.dense(z_sample,4*4*64,activation=None,trainable=isTrainable,reuse=reuse,name='dec_dense_fc_first_layer',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev));
        z_sample = tf.layers.batch_normalization(z_sample,training=isTrainable,reuse=reuse,name='bn_0');
        z_sample = tf.nn.relu(z_sample);
        z_sample = tf.reshape(z_sample,[-1,4,4,64]);
        #7x7x128

        deconv1 = tf.layers.conv2d_transpose(z_sample,kernel_initializer=tf.random_normal_initializer(stddev=stddev),filters=64,kernel_size=[3,3],padding='SAME',activation=None,strides=(2,2),name='dec_deconv1_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv1 = tf.layers.batch_normalization(deconv1,training=isTrainable,reuse=reuse,name='bn_1');
        deconv1 = tf.nn.relu(deconv1,name='relu_deconv_1');
         
        # 14x14x64
        deconv2 = tf.layers.conv2d_transpose(deconv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=32,kernel_size=[3,3],padding='SAME',activation=None,strides=(2,2),name='dec_deconv2_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv2 = tf.layers.batch_normalization(deconv2,training=isTrainable,reuse=reuse,name='bn_2');
        deconv2 = tf.nn.relu(deconv2,name='relu_deconv_2');
        
        #28x28x32
        deconv3 = tf.layers.conv2d_transpose(deconv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=1,kernel_size=[3,3],padding='SAME',activation=None,strides=(2,2),name='dec_deconv3_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv3 = tf.layers.batch_normalization(deconv3,training=isTrainable,reuse=reuse,name='bn_3');
        deconv3 = tf.nn.relu(deconv3,name='relu_deconv_3');
        
        deconv_3_reshaped = tf.layers.flatten(deconv3);

        #28x28x1
        final_op = tf.layers.dense(deconv_3_reshaped,784,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),activation=tf.nn.sigmoid,name='dec_final_layer',trainable=isTrainable,reuse=reuse);
        
        final_op = tf.reshape(final_op,[-1,784]);
        return final_op;

def code_discriminator(Z,isTrainable=True,reuse=False,name='gamma_code_discriminator'):
    with tf.variable_scope(name) as scope:

        if reuse:
            scope.reuse_variables();

        #kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),S

        fc1 = tf.layers.dense(Z,512,activation=None,name='code_dis_fc_layer_1',trainable=isTrainable,reuse=reuse);
        #fc1 = tf.layers.batch_normalization(fc1,training=isTrainable,reuse=reuse,name='bn_1');
        fc1 = tf.nn.relu(fc1);

        fc2 = tf.layers.dense(fc1,512,activation=None,name='code_dis_fc_layer_2',trainable=isTrainable,reuse=reuse);
        #fc2 = tf.layers.batch_normalization(fc2,training=isTrainable,reuse=reuse,name='bn_2');
        fc2 = tf.nn.relu(fc2);

        fc3 = tf.layers.dense(fc2,512,activation=None,name='code_dis_fc_layer_3',trainable=isTrainable,reuse=reuse);
        #fc3 = tf.layers.batch_normalization(fc3,training=isTrainable,reuse=reuse,name='bn_3');
        fc3 = tf.nn.relu(fc3);

        logits = tf.layers.dense(fc3,1,activation=None,name='code_dis_fc_layer_4',trainable=isTrainable,reuse=reuse);

        return logits;
