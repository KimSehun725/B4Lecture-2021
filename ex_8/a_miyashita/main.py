import argparse
import sys
import time

from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from matplotlib import pyplot as plt


class Hmm:
    def __init__(self, k=None, m=None, pi=None, a=None, b=None):
        if pi is None:
            if k is None:
                sys.exit("Hmm.__init__:the number of status is required")
            self.pi = np.zeros(k)
        else:
            self.pi = pi
        
        if a is None:
            if k is None:
                sys.exit("Hmm.__init__:the number of status is required")
            self.a = np.zeros((k,k))
        else:
            self.a = a

        if b is None:
            if k is None:
                sys.exit("Hmm.__init__:the number of status is required")
            if m is None:
                sys.exit("Hmm.__init__:the number of outputs is required")
            self.b = np.zeros((k,m))
        else:
            self.b = b

        self.k = self.pi.shape[0]
        self.m = self.b.shape[1]

    def forward(self, x, scaling=False):
        # (k,samples,t)
        pout = self.b[:,x]
        # (k,samples,t)
        alpha = np.zeros((self.k,)+x.shape)
        # (samples,t)
        c = np.zeros(x.shape)
        # (k,samples)<-(k,samples)*(k,1)
        alpha[:,:,0] = pout[:,:,0]*self.pi[:,np.newaxis]

        if scaling:
            # (samples,)<-(k,samples)
            c[:,0] = np.sum(alpha[:,:,0],axis=0)
            # (k,samples)<-(k,samples)/(1,samples)
            alpha[:,:,0] /= c[np.newaxis,:,0]

        for t in range(1,x.shape[1]):
            # (k,k',samples)<-(k,k',1)*(k,1,samples)
            ave = self.a[:,:,np.newaxis]*alpha[:,np.newaxis,:,t-1]
            # (k',samples)<-(k,k',samples)
            ave = np.sum(ave, axis=0)
            # (k',samples)<-(k',samples)*(k',samples)
            alpha[:,:,t] = pout[:,:,t]*ave

            if scaling:
                # (samples,)<-(k,samples)
                c[:,t] = np.sum(alpha[:,:,t],axis=0)
                # (k,samples)<-(k,samples)/(1,samples)
                alpha[:,:,t] /= c[np.newaxis,:,t]

        return alpha, c

    def backward(self, x, c=None):
        # (k,samples,t)
        pout = self.b[:,x]
        # (k,samples,t)
        beta = np.zeros((self.k,)+x.shape)
        # (k,samples)<-()
        beta[:,:,-1] = 1.
        for t in range(-2,-x.shape[1]-1,-1):
            # (k,k',samples)<-(k,k',1)*(1,k',samples)
            ave = self.a[:,:,np.newaxis]*beta[np.newaxis,:,:,t+1]
            # (k,k',samples)<-(1,k',samples)*(k,k',samples)
            ave = pout[np.newaxis,:,:,t+1]*ave
            # (k,samples)<-(k,k',samples)
            beta[:,:,t] = np.sum(ave, axis=1)

            if c is not None:
                # (k,samples)<-(k,samples)/(1,samples)
                beta[:,:,t] /= c[np.newaxis,:,t+1]

        return beta

    def viterbi(self, x):
        samples = x.shape[0]
        time = x.shape[1]
        
        # (k,samples,t)
        lp = np.log(self.b[:,x])
        # (k,K',1)
        la = np.log(self.a[:,:,np.newaxis])
        # (k,samples,t)
        w = np.zeros((self.k,)+x.shape,dtype=np.int)
        # (k,samples)<-(k,samples)*(k,1)
        nu = lp[:,:,0]+np.log(self.pi[:,np.newaxis])
        # (k,samples)<-(k,1)
        w[:,:,0] = np.arange(self.k)[:,np.newaxis]

        j1 = np.arange(self.k)
        j3 = np.arange(samples)
        jj1,jj2 = np.meshgrid(j1,j3,indexing='ij')

        for t in range(1,time):
            # (k,k',samples)<-(k,k',1)*(k,1,samples)
            cand = la+nu[:,np.newaxis,:]
            # (k',samples)<-(k,k',samples)
            maxidx = np.argmax(cand,axis=0)
            # (k',samples)<-(k',samples)*(k',samples)
            nu = lp[:,:,t]+cand[maxidx,jj1,jj2]
            # (k',samples,t)<-(k,samples,t)
            w = w[maxidx,jj2,:]
            # (k',samples)<-(k',1)
            w[:,:,t] = np.arange(self.k)[:,np.newaxis]

        # (samples)<-(k,samples)
        maxidx = np.argmax(nu,axis=0)
        # (samples)
        lpmax = nu[maxidx,j3]
        # (samples,t)
        zmax = w[maxidx,j3,:]
        
        return lpmax, zmax

    def fit(self, x, y=None):
        # (m,samples,t)
        x_oh = np.eye(self.m)[:,x]
        
        # initialize parameters
        self.pi = 1-np.random.rand(self.k)
        self.pi /= np.sum(self.pi)
        self.a = 1-np.random.rand(self.k,self.k)
        self.a /= np.sum(self.a,axis=1,keepdims=True)
        self.b = np.sum(x_oh,axis=(1,2))
        self.b /= np.sum(self.b)
        self.b = np.broadcast_to(self.b,(self.k,self.m))

        prev = -np.inf
        lp = np.array([])
        
        while 1:
            # E step
            # (k,samples,t),(samples,t)
            alpha,c = self.forward(x,scaling=True)
            # alpha,c = self.forward(x)

            # check
            plaus = np.sum(np.log(c))
            # px = np.sum(alpha[:,:,:,-1],axis=1)
            # plaus = np.sum(np.log(px)*y_oh)
            if plaus - prev < 1e-6:
                if plaus < prev:
                    sys.exit("Hmm.fit:plausibility decreased")
                break
            prev = plaus
            lp = np.append(lp,plaus)

            # (k,samples,t)
            pout = self.b[:,x]
        
            # (k,samples,t)
            beta = self.backward(x,c)
            # beta = self.backward(x)
            # (k,samples,t)<-(k,samples,t)*(k,samples,t)
            gamma = alpha*beta
            # gamma = alpha*beta/px[:,np.newaxis,:,np.newaxis]
            # (k,k',samples,t)<-(k,1,samples,t)*(1,k',samples,t)
            xi = alpha[:,np.newaxis,:,:-1]*beta[:,:,1:]
            # (k,k',samples,t)<-(k,k',samples,t)*(k,k',1,1)*(1,k',samples,t)
            xi *= self.a[:,:,np.newaxis,np.newaxis]*pout[:,:,1:]
            # (k,k',samples,t)<-(k,k',samples,t)/(1,1,samples,t)
            xi /= c[:,1:]
            # xi /= px[:,np.newaxis,np.newaxis,:,np.newaxis]

            # M step
            # (k,)<-(k,sample)
            self.pi = np.sum(gamma[:,:,0],axis=1)
            # (k,)<-(k,)/(1,)
            self.pi /= np.sum(self.pi)
            # (k,k')<-(k,k',samples,t)
            self.a = np.sum(xi,axis=(2,3))
            # (k,k')<-(k,k')/(k,1)
            self.a /= np.sum(self.a,axis=1,keepdims=True)
            # (k,m,samples,t)<-(k,1,samples,t)*(1,m,samples,t)
            # (k,m)<-(k,m,samples,t)
            self.b = np.sum(gamma[:,np.newaxis]*x_oh,axis=(2,3))
            # (k,m)<-(k,m)/(k,1)
            self.b /= np.sum(self.b,axis=1,keepdims=True)
            
        # plt.plot(lp)
        # plt.show()
    
    def predict(self, x):
        """
        alpha, c = self.forward(x)
        beta = self.backward(x)
        # (n,samples)<-(n,k,samples)
        px = np.sum(alpha[:,:,:,0]*beta[:,:,:,0],axis=1)
        
        return np.argmax(px,axis=0)
        """
        alpha, c = self.forward(x, scaling=True)
        # (samples)<-(samples,t)
        lpx = np.sum(np.log(c),axis=1)
        
        return lpx

    def sampling(self, samples, time):
        x = np.zeros((samples,time),dtype=np.int)
        z = np.zeros((samples,time),dtype=np.int)
        s = np.zeros((samples,self.k),dtype=np.int)
        idx = np.arange(samples,dtype=np.int)
        status = np.arange(self.k,dtype=np.int)
        outputs = np.arange(self.m,dtype=np.int)
        labels = np.zeros(samples,dtype=np.int)

        labels = np.random.choice(status,size=samples,p=self.pi)
        for j in range(self.k):
            s[:,j] = np.random.choice(outputs,size=samples,p=self.b[j])
        x[idx,0] = s[idx,labels]
        z[:,0] = labels

        for t in range(1,time):
            for j in range(self.k):
                s[:,j] = np.random.choice(status,size=samples,p=self.a[j])
            labels = s[idx,labels]
            for j in range(self.k):
                s[:,j] = np.random.choice(outputs,size=samples,p=self.b[j])
            x[idx,t] = s[idx,labels]
            z[:,t] = labels
        
        return x, z

    def log_plaus(self, x, z):
        # (samples)
        pini = self.pi[z[:,0]]
        # (samples,t-1)
        ptrans = self.a[z[:,:-1],z[:,1:]]
        # (samples,t)
        pout = self.b[z,x]
        
        lp = np.log(pini)
        lp += np.sum(np.log(ptrans),axis=1)
        lp += np.sum(np.log(pout),axis=1)

        return lp
    

def main():
    # process args
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("sc", type=str, help="input filename with extension .pickle")
    args = parser.parse_args()

    print(f">>> data = pickle.load(open({args.sc}, 'rb'))")
    data = pickle.load(open(args.sc, 'rb'))
    print(">>> print(data['answer_models'].shape)")
    print(data['answer_models'].shape)
    print(">>> print(np.array(data['output']).shape)")
    print(np.array(data['output']).shape)
    print(">>> print(np.array(data['models']['PI']).shape)")
    print(np.array(data['models']['PI']).shape)
    print(">>> print(np.array(data['models']['A']).shape)")
    print(np.array(data['models']['A']).shape)
    print(">>> print(np.array(data['models']['B']).shape)")
    print(np.array(data['models']['B']).shape)

    answer = np.array(data['answer_models'])
    output = np.array(data['output'])
    pi = np.array(data['models']['PI'])
    a = np.array(data['models']['A'])
    b = np.array(data['models']['B'])

    modelsize = pi.shape[0]
    samplesize = output.shape[0]
    models = []
    
    for i in range(modelsize):
        models.append(Hmm(pi=pi[i,:,0],a=a[i],b=b[i]))

    lps = np.zeros((modelsize,samplesize))
    for i in range(modelsize):
        lps[i] = models[i].predict(output)
    
    pred = np.argmax(lps,axis=0)
    labels = np.arange(modelsize)
    cm = confusion_matrix(answer,pred,labels=labels)
    print(cm)
    
    model = Hmm(pi=pi[0,:,0],a=a[0],b=b[0])
    x,z = model.sampling(50,100)
    lpxzpred,zpred = model.viterbi(x)
    lpxz = model.log_plaus(x,z)
    print(lpxz<=lpxzpred)
    labels = np.arange(model.k)
    cm = confusion_matrix(z.ravel(),zpred.ravel(),labels=labels)
    print(cm)
    
    start = time.time()
    for i in range(modelsize):
        models[i].fit(output[answer==i])
    print(time.time()-start)

    lps = np.zeros((modelsize,samplesize))
    for i in range(modelsize):
        lps[i] = models[i].predict(output)
    
    pred = np.argmax(lps,axis=0)
    labels = np.arange(modelsize)
    cm = confusion_matrix(answer,pred,labels=labels)
    print(cm)
    

if __name__=="__main__":
    main()