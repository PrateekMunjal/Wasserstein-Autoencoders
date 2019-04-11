import tensorflow as tf 
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt 
import config

stddev = 0.02;
opts = config.config_cifar10

z_dim = opts['z_dim'];
img_height = opts['img_height'];
img_width = opts['img_width'];
num_channels = opts['num_channels'];

def encoder(X,isTrainable=True,reuse=False,name='phi_encoder'):
    with tf.variable_scope(name) as scope:
        #encoder_activations = {};
        if reuse:
            scope.reuse_variables();

        #32x32x3 --> means size of input before applying conv1
        conv1 = tf.layers.conv2d(X,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv1_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv1 = tf.layers.batch_normalization(conv1,training=isTrainable,reuse=reuse,name='bn_1');
        conv1 = tf.nn.relu(conv1,name='leaky_relu_conv_1');
        
        #16x16x64
        conv2 = tf.layers.conv2d(conv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv2_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv2 = tf.layers.batch_normalization(conv2,training=isTrainable,reuse=reuse,name='bn_2');
        conv2 = tf.nn.relu(conv2,name='leaky_relu_conv_2');
        
        #8x8x128
        conv3 = tf.layers.conv2d(conv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv3_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv3 = tf.layers.batch_normalization(conv3,training=isTrainable,reuse=reuse,name='bn_3');
        conv3 = tf.nn.relu(conv3,name='leaky_relu_conv_3');
    
        #4x4x256
        conv4 = tf.layers.conv2d(conv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=512,kernel_size=[5,5],padding='SAME',strides=(1,1),name='enc_conv4_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv4 = tf.layers.batch_normalization(conv4,training=isTrainable,reuse=reuse,name='bn_4');
        conv4 = tf.nn.relu(conv4,name='leaky_relu_conv_4');
        
        #4x4x512
        conv4_flattened = tf.layers.flatten(conv4);
        
        latent_code = tf.layers.dense(conv4_flattened,z_dim,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),activation=None,name='enc_latent_space',trainable=isTrainable,reuse=reuse);
        return latent_code;

def generator(z_sample,isTrainable=True,reuse=False,name='theta_generator'):
    with tf.variable_scope(name) as scope:  
        #decoder_activations = {};
        if reuse:
            scope.reuse_variables();

        z_sample = tf.layers.dense(z_sample,4*4*512,activation=None,trainable=isTrainable,reuse=reuse,name='dec_dense_fc_first_layer',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev));
        z_sample = tf.layers.batch_normalization(z_sample,training=isTrainable,reuse=reuse,name='bn_0');
        z_sample = tf.nn.relu(z_sample);
        z_sample = tf.reshape(z_sample,[-1,4,4,512]);
        #4x4x512

        deconv1 = tf.layers.conv2d_transpose(z_sample,kernel_initializer=tf.random_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv1_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv1 = tf.layers.batch_normalization(deconv1,training=isTrainable,reuse=reuse,name='bn_1');
        deconv1 = tf.nn.relu(deconv1,name='relu_deconv_1');
         
        #8x8x256
        deconv2 = tf.layers.conv2d_transpose(deconv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv2_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv2 = tf.layers.batch_normalization(deconv2,training=isTrainable,reuse=reuse,name='bn_2');
        deconv2 = tf.nn.relu(deconv2,name='relu_deconv_2');
        
        #16x16x128
        deconv3 = tf.layers.conv2d_transpose(deconv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv3_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv3 = tf.layers.batch_normalization(deconv3,training=isTrainable,reuse=reuse,name='bn_3');
        deconv3 = tf.nn.relu(deconv3,name='relu_deconv_3');
        
        #32x32x64 
        deconv4 = tf.layers.conv2d_transpose(deconv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=3,kernel_size=[5,5],padding='SAME',activation=None,strides=(1,1),name='dec_deconv4_layer',trainable=isTrainable,reuse=reuse); # 16x16    
        #deconv4 = tf.layers.dropout(deconv4,rate=keep_prob,training=True);
        deconv4 = tf.nn.tanh(deconv4);
        #64x64x3
        
        deconv_4_reshaped = tf.reshape(deconv4,[-1,img_height,img_width,num_channels]);
        return deconv_4_reshaped;

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
