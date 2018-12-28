import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mb_size = 64
#z_dim = 100  #latent variable size
z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ====================================== encoder

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1) #(batch,128)

    #z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu  #(batch,100)
    #z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma   #(batch,100)
    a = tf.nn.softplus(tf.matmul(h, Q_W2_mu) + Q_b2_mu) #(batch,100)
    b = tf.nn.softplus(tf.matmul(h, Q_W2_sigma) + Q_b2_sigma)   #(batch,100)

    #u = tf.random(tf.stack([tf.shape(X)[0], z_dim]))
   # v = (1-u**(1/b))**(1/a)  #(batch*100)

    a = tf.cast(a,tf.float32)
    b = tf.cast(b,tf.float32)
    return a, b


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))

    return mu + tf.exp(log_var / 2) * eps


def sample_sb_np(a,b,mb_size):
    
    u = np.random.rand(mb_size, z_dim)
    v = (1-u**(1/b))**(1/a)  #(batch*100)

    #pi = tf.Variable(tf.ones(shape=(mb_size,z_dim)))
    #pi = np.ones((mb_size,z_dim))
    #pi_list = [tf.slice(v, [0, 0], [mb_size, 1])]
    pi = np.ones((mb_size,z_dim))
    pi[:,0] = v[:,0]
    for i in range(1,z_dim):
        #pi_ = np.ones((mb_size,1))
        for j in range(0,i):
            #v_j = tf.slice(v, [0, j], [mb_size, 1])
            v_j = v[:,j]
            pi[:,i] = pi[:,i]*(1-v_j)
        #v_i = tf.slice(v, [0, i], [mb_size, 1])
        v_i = v[:,i]
        pi[:,i] = pi[:,i]*v_i
    return pi

def sample_sb(a,b,mb_size):
    print(a)
    print(b)

    tf.random.uniform(shape = tf.shape(a),minval = 0.01,maxval = 0.99)
    #u = np.random.randn(mb_size, z_dim)
    v = (1-u**(1/b))**(1/a)  #(batch*100)

    #pi = tf.Variable(tf.ones(shape=(mb_size,z_dim)))
    #pi = np.ones((mb_size,z_dim))
    pi_list = [tf.slice(v, [0, 0], [mb_size, 1])]

    for i in range(1,z_dim):
        pi_ = tf.Variable(tf.ones(shape=(mb_size,1)))
        for j in range(0,i):

            v_j = tf.slice(v, [0, j], [mb_size, 1])

            #pi[:,i] = pi[:,i]* (1-v[:,j]) #operating on tensor should be right
            pi_ = pi_*(1-v_j)

        v_i = tf.slice(v, [0, i], [mb_size, 1])
        
        '''
        print("pi_",pi_)
        print("v_i",v_i)
        print("i",i)
        print("j",j)
        '''
        pi_ = pi_*v_i
        pi_list.append(pi_)
        #pi[:,i] = pi[:,i]*v[:,i]
    pi = tf.concat(pi_list,1)
    #pi = tf.convert_to_tensor(pi)
    return pi
# =============================== P(X|z) ======================================  decoder

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)

    return prob, logits

#-----------------------KL--------------------------
def Beta_fn(a, b):
    print("Beta a",a)
    print("Beta b",b)

    return tf.math.exp(tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a+b))

def compute_KL(prior_alpha, prior_beta,a,b):#why - default Beta(1,5)
    #kl = 0
    #for i in range():
    #    kl = 1./(i+a*b) * Beta_fn(i/a, b)

    #Between K distribution and Beta(a,b)
    epsilon = 0.000001
    print("KL a",a)  #print 直接得到 tensor的type
    print("KL b",b)

    prior_alpha = tf.Variable(tf.ones(shape=(mb_size,z_dim))) * prior_alpha
    prior_beta = tf.Variable(tf.ones(shape=(mb_size,z_dim))) * prior_beta


    kl = 1./(1.+a*b) * Beta_fn(1./a, b)
    kl += 1./(2.+a*b) * Beta_fn(2./a, b)
    kl += 1./(3.+a*b) * Beta_fn(3./a, b)
    kl += 1./(4.+a*b) * Beta_fn(4./a, b)
    kl += 1./(5.+a*b) * Beta_fn(5./a, b)
    kl += 1./(6.+a*b) * Beta_fn(6./a, b)
    kl += 1./(7.+a*b) * Beta_fn(7./a, b)
    kl += 1./(8.+a*b) * Beta_fn(8./a, b)
    kl += 1./(9.+a*b) * Beta_fn(9./a, b)
    kl += 1./(10.+a*b) * Beta_fn(10./a, b)
    kl *= (prior_beta-1.)*b

     # use another taylor approx for Digamma function                                                                                                                                             
    psi_b_taylor_approx = tf.math.log(b + epsilon) - 1./(2. * b) - 1./(12. * b**2)
    kl += (a-prior_alpha)/a * (-0.57721 - psi_b_taylor_approx - 1./b) #T.psi(self.posterior_b)                                                                                        

    # add normalization constants                                                                                                                                                                
    kl += tf.math.log(a*b+epsilon) + tf.math.log(Beta_fn(prior_alpha, prior_beta) + epsilon)  #做出来和a,b一样大的tf.constant来

    # final term                                                                                                                                                                                 
    kl += -(b-1)/b

    return kl
    #return tf.reduce_sum(kl , 1)



# =============================== TRAINING ====================================

a, b = Q(X)

#z_sample = sample_z(z_mu, z_logvar)

#z_sample= sample_sb(a, b, mb_size)  #stick breaking 

#-------------sample_sb-----------------------
#a b ok

u = tf.random.uniform(shape = tf.shape(a),minval = 0.01,maxval = 0.99)

v = (1-u**(1/b))**(1/a)  #(batch*100)
pi_list = [tf.slice(v, [0, 0], [mb_size, 1])]

for i in range(1,z_dim):
    pi_ = tf.Variable(tf.ones(shape=(mb_size,1)))
    for j in range(0,i):
        v_j = tf.slice(v, [0, j], [mb_size, 1])
        pi_ = pi_*v_j
    v_i = tf.slice(v, [0, i], [mb_size, 1])
    pi_ = pi_*v_i
    pi_list.append(pi_)

z_sample = tf.concat(pi_list,1)

#-------------------------------------------

_, logits = P(z_sample)

# Sampling from random z
X_samples, _ = P(z)

# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)

# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian

#kl_loss = 0.5 * tf.reduce_sum(tf.exp(b) + a**2 - 1. - b, 1)  #Gaussian KL divergence  (batch,1)

#K分布的ab 和Beta分布的(alpha,beta)有什么关系
kl_loss_list = compute_KL(1,5,a,b)

greater_equal=tf.greater_equal(kl_loss_list , 0)
kl_loss = tf.reduce_sum(kl_loss_list, 1)
  #stick breaking KL divergence
# VAE loss
kl_loss_mean = tf.reduce_mean(kl_loss)
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    #_, loss = sess.run([solver, vae_loss,], feed_dict={X: X_mb})
    #a,b = sess.run([a,b], feed_dict={X: X_mb})
    #a = a.eval() # tensor to numpy
    #b = b.eval()

    #_, loss, a_ = sess.run([solver, vae_loss,a], feed_dict={X: X_mb})
    _, loss,kl_loss_list_,recon_loss_, a_, b_,z_sample_,logits_,v_,u_,kl_loss_mean_,greater_equal_ = sess.run([solver, vae_loss,kl_loss_list,recon_loss,a,b,z_sample,logits,v,u,kl_loss_mean,greater_equal], feed_dict={X: X_mb})

    #_, loss= sess.run([solver, vae_loss], feed_dict={a: a, b: b, mb_size: mb_size})

    #if it % 1000 == 0:
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))

        print('a:',a_)
        print('b:',b_)
        print("v", v_)
        print("u",u_)
        print('z_sample',z_sample_)
        #print('logits',logits_)

        print("kl_loss_list",kl_loss_list_)
        print("kl loss mean", kl_loss_mean_)
        print("recon_loss", recon_loss_)

        print("greater_equal",greater_equal_)
        print()

        feed_in = {z: sample_sb_np(1,5,16)}
        #samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})
        samples = sess.run(X_samples, feed_dict=feed_in)

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
