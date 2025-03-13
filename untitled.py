import numpy as np
import random
import matplotlib.pyplot as plt
import collections
from itertools import product

def generate_H(n,c1=0,c2=0):
    A= np.ones((n,3)) # 1,file size 2,amount of computaion 3,time delay tolerance
    A[:,0]=np.random.randint(12,16,size=n)+c1 #si Mega
    A[:,1]=np.random.randint(2000,2500,size=n)+c2 #ci Mega
    A[:,2]=np.ones(n)*3
    return A




def MakeDspace(n,step):

    array=[0,1]
    eta=np.array(list(product(array,repeat=n)))
    
    array2=np.round(np.arange(0, 1+step, step),4)
    theta=np.array(list(product(array2,repeat=n)))
    
    array3=np.round(np.arange(0, 1+step, step),4)
    beta=np.array(list(product(array3,repeat=n)))
    q=np.empty((0,3,n))
    for i in range(eta.shape[0]):
        e=np.array(eta[i]).reshape(-1,n)
        t=np.unique(e*theta[np.sum(e*theta,axis=1)<=1],axis=0)
        t=np.unique(e*t[np.sum(np.abs(e-np.ceil(t)),axis=1)==0],axis=0)
        b=np.unique(e*beta[np.sum(e*beta,axis=1)<=1],axis=0)   
        b=np.unique(e*b[np.sum(np.abs(e-np.ceil(b)),axis=1)==0],axis=0)      
        if t.shape[0]*b.shape[0]!=0:
            tb=np.array(list(product(t,b)))
            p=np.zeros((t.shape[0]*b.shape[0],3,n))
            p[:,0,:]=e
            p[:,1:3,:]=tb
        q=np.append(q,p,axis=0)
    return q
    
    
def MakeZspace(step):
    E=np.array([0])
    comp=np.arange(0, 1+step, step)
    Z=np.array(list(product(E,comp)))
    return Z


    
def calZ(H,D,timestep):
    L=200
    p= 1.2 # trpower for uploading
    pw=0.8
    g= 127+30*np.log(L) # channel gain 
    W= 5*10**6 #bandwith
    noisedbm= -100 # power of WGN
    noise=10**(noisedbm/10)*10**(-3)
    R= W*np.log2(1+p*g/(noise**0.5)) # upload rate
    f_loc=0.8*10**9
    f_mec=6*10**9
    energy=0
    Time=0
    
    #local
    energy +=(1-D[0])*H[:,1]*10**6*(f_loc**2)*(10**(-27))
    Time += (1-D[0])*H[:,1]*10**6/f_loc
    #MEC
    energy +=D[0]*p*H[:,0]*10**6/(R*(1-D[0]+D[1])) #trans
    Time += D[0]*H[:,0]*10**6/(R*(1-D[0]+D[1]))
    energy +=D[0]*pw*H[:,1]*10**6/((1-D[0]+D[2])*f_mec) #calculate
    Time +=D[0]*H[:,1]*10**6/((1-D[0]+D[2])*f_mec)
    #check state
    z= np.zeros(2)
    z[0]=np.sum(energy)
    #aa=Time>(timestep*0.8)
    #z[1]=1-np.sum(aa*D[2])
    z[1]=1-np.sum(D[0]*D[2])
    return z,Time








def createEpsilonGreedyPolicy(Q, epsilon, num_actions):

    def policyFunction(state):
   
        Action_probabilities = np.ones(num_actions,dtype = float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities
   
    return policyFunction
    
def Qlearning(Z,D,lr,dfactor,n,timestep,c1=0,c2=0):
    


    Q = collections.defaultdict(lambda:np.zeros(D.shape[0]))
    epsilon=0.01

    #for each episode
    episode = Z
    policy = createEpsilonGreedyPolicy(Q, epsilon, D.shape[0])
    zstep=(Z[1]-Z[0])[1]
    for j in range(2*D.shape[0]):   

        for i in range(episode.shape[0]):
            ti=0
            next_z=None
            z_l=np.array([])
            q_l=np.array([])
            nz=i
            totalT=0
            H=generate_H(n,c1,c2)
            h= H*timestep/H[0][2]
            while ti<H[0][2]:
                actionprob=policy(nz)
                ti+=timestep
                zt=Z[nz]
                avaD= (np.sum(D,axis=2)[:,2]<=zt[1])
                action = np.random.choice(np.arange(len(actionprob)),p = (actionprob*avaD)/sum(actionprob*avaD))
                
                
                next_z, T= calZ(h,D[action],timestep)
                next_z_i= np.where((next_z[1]-0.2<Z[:,1])&(Z[:,1]<=next_z[1]))[0][0]
                reward=-next_z[0]

                avaDnext=(np.sum(D,axis=2)[:,2]<=episode[next_z_i][1])

                best_next_action = np.argmax(Q[next_z_i]-1000*(1-avaDnext))    


                td_target = reward + dfactor * Q[next_z_i][best_next_action]
                td_delta = td_target - Q[nz][action]
                Q[nz][action] += lr * td_delta          
                z_l=np.append(z_l,nz)
                q_l=np.append(q_l,action)
                totalT+=T
                nz= next_z_i
            if np.any(totalT>=H[0][2]):
                newre= np.max(totalT)-H[0][2]
                for ii in range(len(z_l)):
                    Q[z_l[ii]][int(q_l[ii])]-=lr*(newre)*100

    return Q





def simulql_n(timestep,lr,step,dfactor):
    TL=0
    zi=0
    EL=0
    n=5
    Z=MakeZspace(step)
    
    TL=0
    EL=0
    Eresult= np.zeros(n)
    Tresult= np.zeros(n)
    
    TL_off=0
    EL_off=0    
    Eresult_off= np.zeros(n)
    Tresult_off= np.zeros(n)
    
    TL_lo=0
    EL_lo=0
    Eresult_lo= np.zeros(n)
    Tresult_lo= np.zeros(n)
    
    c1=0
    c2=0
    for j in range(n):
        D=MakeDspace(j+1,step)
        Q=Qlearning(Z,D,lr,dfactor,j+1,timestep,c1,c2)
        TL=0
        EL=0
        TL_off=0
        EL_off=0
        TL_lo=0
        EL_lo=0
        zi=0
        D_off=np.zeros((3,j+1))
        D_off[0,:]+=1
        D_off[1,:]+=1/(j+1)
        D_off[2,:]+=1/(j+1)
        D_lo=np.zeros((3,j+1))
        H=generate_H(j+1,c1,c2)
        for i in range(int(3/timestep)):
            h= H*timestep/H[0][2]
            avd= (np.sum(D,axis=2)[:,2]<=Z[zi][1])
            action=np.argmax(Q[zi]-100*(1-avd))
            next_z,T=calZ(h,D[action],timestep)
            TL+=np.average(T)
            EL+=next_z[0]
            zi=np.where((next_z[1]-0.2<Z[:,1])&(Z[:,1]<=next_z[1]))[0][0]
           
      
            z_lo,T_lo=calZ(h,D_lo,timestep)
            TL_lo+=np.average(T_lo)
            EL_lo+=z_lo[0]

            z_off,T_off=calZ(h,D_off,timestep)
            TL_off+=np.average(T_off)
            EL_off+=z_off[0] 

          
                        
            

            
        Eresult[j]=EL
        Tresult[j]=TL
        
        Eresult_off[j]=EL_off
        Tresult_off[j]=TL_off
        
        Eresult_lo[j]=EL_lo
        Tresult_lo[j]=TL_lo
    
    plt.plot(list(range(1,n+1)),Eresult,label='Q')
    plt.plot(list(range(1,n+1)),Eresult_lo,label='local')
    plt.plot(list(range(1,n+1)),Eresult_off,label='offloading')
    plt.legend()
    plt.title(f'energy lr={lr}, dfactor={dfactor}')
    plt.xticks(list(range(1,n+1)))
    plt.xlabel('number of UEs')
    plt.ylabel('Energy consumption(J)')
    plt.show()
    
    
    
    plt.plot(list(range(1,n+1)),Tresult,label='Q')
    plt.plot(list(range(1,n+1)),Tresult_lo,label='local')
    plt.plot(list(range(1,n+1)),Tresult_off,label='offloading')
    plt.legend()
    plt.title(f'time lr={lr}, dfactor={dfactor}')
    plt.xlabel('number of UEs')
    plt.ylabel('average time(s)')
    plt.xticks(list(range(1,n+1)))
    plt.show()
    
def simulql_n1(timestep,lr,step,dfactor):
    TL=0
    zi=0
    EL=0
    n=4
    Z=MakeZspace(step)
    
    TL=0
    EL=0
    Eresult= np.zeros(5)
    Tresult= np.zeros(5)
    
    TL_off=0
    EL_off=0    
    Eresult_off= np.zeros(5)
    Tresult_off= np.zeros(5)
    
    TL_lo=0
    EL_lo=0
    Eresult_lo= np.zeros(5)
    Tresult_lo= np.zeros(5)
    
    c1l=[-12,-8,-4,0,4]
    c2l=[-1500,-1000,-500,0,500]
    for j in range(len(c1l)):
        c1=c1l[j]
        c2=c2l[j]
        D=MakeDspace(n,step)
        Q=Qlearning(Z,D,lr,dfactor,n,timestep,c1,c2)
        TL=0
        EL=0
        TL_off=0
        EL_off=0
        TL_lo=0
        EL_lo=0
        zi=0
        D_off=np.zeros((3,n))
        D_off[0,:]+=1
        D_off[1,:]+=1/(n)
        D_off[2,:]+=1/(n)
        D_lo=np.zeros((3,n))
        H=generate_H(n,c1,c2)
        for i in range(int(3/timestep)):
            h= H*timestep/H[0][2]
            avd= (np.sum(D,axis=2)[:,2]<=Z[zi][1])
            action=np.argmax(Q[zi]-100*(1-avd))
            next_z,T=calZ(h,D[action],timestep)
            TL+=np.average(T)
            EL+=next_z[0]
            zi=np.where((next_z[1]-0.2<Z[:,1])&(Z[:,1]<=next_z[1]))[0][0]
           
            #if i!=int(3/timestep):
#                z_off,T_off=calZ(h,D_off,timestep)
 #               TL_off+=np.average(T_off)
  #              EL_off+=z_off[0]
   #             TL_off+=timestep*0.8
    #            EL_off+=(j+1)*0.8*timestep*0.8
     #       else:
      #          z_off,T_off=calZ(h,D_off,timestep)
       #         TL_off+=np.average(T_off)
        #        EL_off+=z_off[0]       
            z_off,T_off=calZ(h,D_off,timestep)
            TL_off+=np.average(T_off)
            EL_off+=z_off[0] 
        
        
            z_lo,T_lo=calZ(h,D_lo,timestep)
            TL_lo+=np.average(T_lo)
            EL_lo+=z_lo[0]

#            z_off,T_off=calZ(h,D_off,timestep)
#            TL_off+=np.average(T_off)
#            EL_off+=z_off[0] 
#            if z_off[1]==1:
#                pass
#            else:
#                TL_off+=timestep
#                EL_off+=(n)*0.8*timestep
          
                        
            

            
        Eresult[j]=EL
        Tresult[j]=TL
        
        Eresult_off[j]=EL_off
        Tresult_off[j]=TL_off
        
        Eresult_lo[j]=EL_lo
        Tresult_lo[j]=TL_lo
    
    plt.plot(list(range(1,n+2)),Eresult,label='Q')
    plt.plot(list(range(1,n+2)),Eresult_lo,label='local')
    plt.plot(list(range(1,n+2)),Eresult_off,label='offloading')
    plt.legend()
    plt.title(f'energy lr={lr}, dfactor={dfactor}, UEs=4')
    plt.xticks(list(range(1,n+2)),labels=['500~1000','1000~1500','1500~2000','2000~2500','2500~3000'])
    plt.xlabel('computation cycle of data (Mbits)')
    plt.ylabel('Energy consumption(J)')
    plt.show()
    
    
    
    plt.plot(list(range(1,n+2)),Tresult,label='Q')
    plt.plot(list(range(1,n+2)),Tresult_lo,label='local')
    plt.plot(list(range(1,n+2)),Tresult_off,label='offloading')
    plt.legend()
    plt.title(f'time lr={lr}, dfactor={dfactor},UEs=5')
    plt.xlabel('computation cycle of data (Mbits)')
    plt.ylabel('average time(s)')
    plt.xticks(list(range(1,n+1)),labels=['500~1000','1000~1500','1500~2000','2000~2500','2500~3000'])
    plt.show()

    

    

'''
def DQN(Z,D,lr,dfactor,n,timestep):
    

###QTABLE###
    Q = Qlearning(Z,D,lr,dfactor,n,timestep)
    ##input=stae
    nn_input_dim = 1
    nn_output_dim = D.shape[0]
    nn_hdim1 = 100
    nn_hdim2 = 100
    lr = 0.001 
    L2_norm = 0.001
    epoch = 50000
    model=NeuralNetwork(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim, init="random")
    stats = model.train(Z, Q, learning_rate=lr, L2_norm=L2_norm, epoch=epoch, print_loss=True)
    return Q

class NeuralNetwork(object):
    def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim, init="random"):
        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
        
        Args:
            nn_input_dim: (int) The dimension D of the input data.
            nn_hdim1: (int) The number of neurons  in the hidden layer H1.
            nn_hdim2: (int) The number of neurons H2 in the hidden layer H1.
            nn_output_dim: (int) The number of classes C.
            init: (str) initialization method used, {'random', 'constant'}
        
        Returns:
            
        """
        # reset seed before start
        np.random.seed(0)
        self.model = {}

        if init == "random":
            self.model['W1'] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.random.randn(nn_hdim2, nn_output_dim)
            self.model['b3'] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model['W1'] = np.ones((nn_input_dim, nn_hdim1))
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.ones((nn_hdim1, nn_hdim2))
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.ones((nn_hdim2, nn_output_dim))
            self.model['b3'] = np.zeros((1, nn_output_dim))

    def forward_propagation(self, X):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
        
        Args:
            X: Input data of shape (N, D)
            
        Returns:
            y_hat: (numpy array) Array of shape (N, C) giving the classification scores for X
            cache: (dict) Values needed to compute gradients
            
        """
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        
        ### CODE HERE ###
        h1= np.dot(X,W1) + b1 # 300 10
        z1= sigmoid(h1) # 300 10
        h2= np.dot(z1,W2) + b2 # 300 10
        z2= tanh(h2) # 300 10
        h3= np.dot(z2,W3) + b3 # 300 2
        y_hat = 200*h3
        ############################
        cache = {'h1': h1, 'z1': z1, 'h2': h2, 'z2': z2, 'h3': h3, 'y_hat': y_hat}
    
        return y_hat, cache

    def back_propagation(self, cache, X, y, L2_norm=0.0):
        """
        Compute the gradients
        
        Args:
            cache: (dict) Values needed to compute gradients
            X: (numpy array) Input data of shape (N, D)
            y: (numpy array) One-hot encoding of training labels (N, C)
            L2_norm: (int) L2 normalization coefficient
            
        Returns:
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
            
        """
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        h1, z1, h2, z2, h3, y_hat = cache['h1'], cache['z1'], cache['h2'], cache['z2'], cache['h3'], cache['y_hat']

        ### CODE HERE ###
                
        dL_dh3= -1*(y-y_hat)*200
        dh3_dz2=W3.T
        dz2_dh2=1-z2**2 #(1-tanh(h2)**2)
        dh2_dz1=W2.T
        dz1_dh1=z1*(1-z1)#(1-sigmoid(h1))*sigmoid(h1)
        
        dh3_dw3=z2.T
        dh2_dw2=z1.T
        dh1_dw1=X.T
        
        
        dW3 = np.dot(dh3_dw3,dL_dh3)+2*L2_norm*W3
        db3= np.sum(dL_dh3,axis=0)
        ####
        dL_dz2= np.dot(dL_dh3,dh3_dz2)
        dL_dh2= dL_dz2*dz2_dh2
        
        dW2= np.dot(dh2_dw2,dL_dh2)+2*L2_norm*W2
        db2= np.sum(dL_dh2,axis=0)
        ###
        dL_dz1= np.dot(dL_dh2,dh2_dz1)
        dL_dh1= dL_dz1*dz1_dh1
        
        dW1= np.dot(dh1_dw1,dL_dh1)+2*L2_norm*W1
        db1= np.sum(dL_dh1,axis=0)
        
        
        ############################
        
        grads = dict()
        grads['dW3'] = dW3
        grads['db3'] = db3
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW1'] = dW1
        grads['db1'] = db1

        return grads

    def compute_loss(self, y_pred, y_true, L2_norm=0.0):
        """
        Descriptions:
            Evaluate the total loss on the dataset
        
        Args:
            y_pred: (numpy array) Predicted target (N,C)
            y_true: (numpy array) Array of training labels (N,C)
        
        Returns:
            loss: (float) Loss (data loss and regularization loss) for training samples.
        """
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']

        ### CODE HERE ###
        L=0.5*(y_true-y_pred)**2
        LL= np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2)
        total_loss= L2_norm*LL+L
        ############################

        return total_loss
        

    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=1e-3, L2_norm=0.0, epoch=20000, print_loss=True):
        """
        Descriptions:
            Train the neural network using gradient descent.
        
        Args:
            X_train: (numpy array) training data (N, D)
            X_val: (numpy array) validation data (N, D)
            y_train: (numpy array) training labels (N,)
            y_val: (numpy array) valiation labels (N, )
            y_pred: (numpy array) Predicted target (N,)
            learning_rate: (float) Scalar giving learning rate for optimization
            L2_norm: (float) Scalar giving regularization strength.
            epoch: (int) Number of epoch to take
            print_loss: (bool) if true print loss during optimization

        Returns:
            A dictionary giving statistics about the training process
        """

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        
        

        for it in range(epoch):

            ### CODE HERE ###
            y_pred, cache = self.forward_propagation(X_train)
            grads = self.back_propagation(cache, X_train, y_train, L2_norm=L2_norm)
            loss=self.compute_loss(y_pred, y_train, L2_norm=L2_norm)
            self.model['W1']-=learning_rate*grads['dW1']
            self.model['b1']-=learning_rate*grads['db1']
            self.model['W2']-=learning_rate*grads['dW2']
            self.model['b2']-=learning_rate*grads['db2']
            self.model['W3']-=learning_rate*grads['dW3']
            self.model['b3']-=learning_rate*grads['db3']
            ################# 
            if (it+1) % 1000 == 0:
                loss_history.append(loss)

                y_train_pred = self.predict(X_train)

            if print_loss and (it+1) % 1000 == 0:
                print(f"Loss (epoch {it+1}): {loss}")
 

    def predict(self, X):
        ### CODE HERE ###
        y_hat, cache = self.forward_propagation(X)
        return y_hat
        #################  



def tanh(x):
    ### CODE HERE ###
    x= 2/(1+np.exp(-2*x))-1
   
    #################  
    return x
    

def relu(x):
    ### CODE HERE ###
    x= x if x>0 else 0

    ############################
    return x 


def sigmoid(x):
    ### CODE HERE ###
    x= 1/(1+np.exp(-x))
    
    ############################
    return x
'''
