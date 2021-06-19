import argparse
import sys
from numpy import random

from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from matplotlib import pyplot as plt


class Hmm:
    def __init__(self, n=None, k=None, m=None, pi=None, a=None, b=None):
        if pi is None:
            if n is None:
                sys.exit("Hmm.__init__:the number of models is required")
            if k is None:
                sys.exit("Hmm.__init__:the number of status is required")
            self.pi = np.zeros((n,k))
        else:
            self.pi = pi
        
        if a is None:
            if n is None:
                sys.exit("Hmm.__init__:the number of models is required")
            if k is None:
                sys.exit("Hmm.__init__:the number of status is required")
            self.a = np.zeros((n,k,k))
        else:
            self.a = a

        if b is None:
            if n is None:
                sys.exit("Hmm.__init__:the number of models is required")
            if k is None:
                sys.exit("Hmm.__init__:the number of status is required")
            if m is None:
                sys.exit("Hmm.__init__:the number of outputs is required")
            self.b = np.zeros((n,k,m))
        else:
            self.b = b

        self.n = self.pi.shape[0]
        self.k = self.pi.shape[1]
        self.m = self.b.shape[2]

    def forward(self, x, scaling=False):
        # (n,k,samples,t)
        pout = self.b[:,:,x]
        # (n,k,samples,t)
        alpha = np.zeros((self.n,self.k)+x.shape)
        # (n,samples,t)
        c = np.zeros((self.n,)+x.shape)
        # (n,k,samples)<-(n,k,samples)*(n,k,1)
        alpha[:,:,:,0] = pout[:,:,:,0]*self.pi[:,:,np.newaxis]

        if scaling:
            c[:,:,0] = np.sum(alpha[:,:,:,0],axis=1)
            # (n,k,samples)<-(n,k,samples)/(n,1,samples)
            alpha[:,:,:,0] /= c[:,np.newaxis,:,0]

        for t in range(1,x.shape[1]):
            # (n,k,k',samples)<-(n,k,k',1)*(n,k,1,samples)
            ave = self.a[:,:,:,np.newaxis]*alpha[:,:,np.newaxis,:,t-1]
            # (n,k',samples)<-(n,k,k',samples)
            ave = np.sum(ave, axis=1)
            # (n,k',samples)<-(n,k',samples)*(n,k',samples)
            alpha[:,:,:,t] = pout[:,:,:,t]*ave

            if scaling:
                c[:,:,t] = np.sum(alpha[:,:,:,t],axis=1)
                alpha[:,:,:,t] /= c[:,np.newaxis,:,t]

        return alpha, c

    def backward(self, x, c=None):
        # (n,k,samples,t)
        pout = self.b[:,:,x]
        # (n,k,samples,t)
        beta = np.zeros((self.n,self.k)+x.shape)
        # (n,k,samples)<-()
        beta[:,:,:,-1] = 1.
        for t in range(-2,-x.shape[1]-1,-1):
            # (n,k,k',samples)<-(n,k,k',1)*(n,1,k',samples)
            ave = self.a[:,:,:,np.newaxis]*beta[:,np.newaxis,:,:,t+1]
            # (n,k,k',samples)<-(n,1,k',samples)*(n,k,k',samples)
            ave = pout[:,np.newaxis,:,:,t+1]*ave
            # (n,k,samples)<-(n,k,k',samples)
            beta[:,:,:,t] = np.sum(ave, axis=2)

            if c is not None:
                beta[:,:,:,t] /= c[:,np.newaxis,:,t+1]

        return beta

    def viterbi(self, x):
        samples = x.shape[0]
        time = x.shape[1]
        
        # (n,k,samples,t)
        lp = np.log(self.b[:,:,x])
        # (n,k,K',1)
        la = np.log(self.a[:,:,:,np.newaxis])
        # (n,k,samples,t)
        w = np.zeros((self.n,self.k)+x.shape,dtype=np.int)
        # (n,k,samples)<-(n,k,samples)*(n,k,1)
        nu = lp[:,:,:,0]+np.log(self.pi[:,:,np.newaxis])
        # (n,k,samples)<-(1,k,1)
        w[:,:,:,0] = np.arange(self.k)[np.newaxis,:,np.newaxis]

        j1 = np.arange(self.n)
        j2 = np.arange(self.k)
        j3 = np.arange(samples)
        jj1,jj3 = np.meshgrid(j1,j3,indexing='ij')
        jjj1,jjj2,jjj3 = np.meshgrid(j1,j2,j3,indexing='ij')

        for t in range(1,time):
            # (n,k,k',samples)<-(n,k,k',1)*(n,k,1,samples)
            cand = la+nu[:,:,np.newaxis,:]
            # (n,k',samples)<-(n,k,k',samples)
            maxidx = np.argmax(cand,axis=1)
            # (n,k',samples)<-(n,k',samples)*(n,k',samples)
            nu = lp[:,:,:,t]+cand[jjj1,maxidx,jjj2,jjj3]
            # (n,k',samples,t)<-(n,k,samples,t)
            w = w[jjj1,maxidx,jjj3,:]
            # (n,k',samples)<-(1,k',1)
            w[:,:,:,t] = np.arange(self.k)[np.newaxis,:,np.newaxis]

        # (n,samples)<-(n,k,samples)
        maxidx = np.argmax(nu,axis=1)
        # (n,samples)
        lpmax = nu[jj1,maxidx,jj3]
        # (n,samples,t)
        zmax = w[jj1,maxidx,jj3,:]
        
        return lpmax, zmax

    def fit(self, x, y):
        # (m,samples,t)
        x_oh = np.eye(self.m)[:,x]
        # (n,samples)
        y_oh = np.eye(self.n)[:,y]

        # initialize parameters
        self.pi = 1-np.random.rand(self.n,self.k)
        self.pi /= np.sum(self.pi,axis=1,keepdims=True)
        self.a = 1-np.random.rand(self.n,self.k,self.k)
        self.a /= np.sum(self.a,axis=2,keepdims=True)
        self.b = np.sum(x_oh*y_oh[:,np.newaxis,:,np.newaxis],axis=(2,3))
        self.b /= np.sum(self.b,axis=1,keepdims=True)
        self.b = np.broadcast_to(self.b[:,np.newaxis],(self.n,self.k,self.m))

        prev = -np.inf
        lp = np.array([])
        
        while 1:
            # E step
            # (n,k,samples,t),(n,samples,t)
            alpha,c = self.forward(x,scaling=True)
            # alpha,c = self.forward(x)

            # check
            plaus = np.sum(np.log(c)*y_oh[:,:,np.newaxis])
            # px = np.sum(alpha[:,:,:,-1],axis=1)
            # plaus = np.sum(np.log(px)*y_oh)
            if plaus - prev < 1e-3:
                if plaus < prev:
                    sys.exit("Hmm.fit:plausibility decreased")
                break
            prev = plaus
            lp = np.append(lp,plaus)

            # (n,k,samples,t)
            pout = self.b[:,:,x]
        
            # (n,k,samples,t)
            beta = self.backward(x,c)
            # beta = self.backward(x)
            # (n,k,samples,t)<-(n,k,samples,t)*(n,k,samples,t)
            gamma = alpha*beta
            # gamma = alpha*beta/px[:,np.newaxis,:,np.newaxis]
            # (n,k,k',samples,t)<-(n,k,1,samples,t)*(n,1,k',samples,t)
            xi = alpha[:,:,np.newaxis,:,:-1]*beta[:,np.newaxis,:,:,1:]
            # (n,k,k',samples,t)<-(n,k,k',samples,t)*(n,k,k',1,1)*(n,1,k',samples,t)
            xi *= self.a[:,:,:,np.newaxis,np.newaxis]*pout[:,np.newaxis,:,:,1:]
            # (n,k,k',samples,t)<-(n,k,k',samples,t)/(n,1,1,samples,t)
            xi /= c[:,np.newaxis,np.newaxis,:,1:]
            # xi /= px[:,np.newaxis,np.newaxis,:,np.newaxis]

            # M step
            # (n,k,sample)<-(n,k,sample)*(n,1,sample)
            # (n,k)<-(n,k,sample)
            self.pi = np.sum(gamma[:,:,:,0]*y_oh[:,np.newaxis],axis=2)
            # (n,k)<-(n,k)/(n,1)
            self.pi /= np.sum(self.pi,axis=1,keepdims=True)
            # (n,k,k',samples,t)<-(n,k,k',samples,t)*(n,1,1,samples,1)
            # (n,k,k')<-(n,k,k',samples,t)
            self.a = np.sum(xi*y_oh[:,np.newaxis,np.newaxis,:,np.newaxis],axis=(3,4))
            # (n,k,k')<-(n,k,k')/(n,k,1)
            self.a /= np.sum(self.a,axis=2,keepdims=True)
            # (n,k,m,samples,t)<-(n,k,1,samples,t)*(1,1,m,samples,t)*(n,1,1,samples,1)
            # (n,k,m)<-(n,k,m,samples,t)
            self.b = np.sum(gamma[:,:,np.newaxis]*x_oh*y_oh[:,np.newaxis,np.newaxis,:,np.newaxis],axis=(3,4))
            # (n,k,m)<-(n,k,m)/(n,k,1)
            self.b /= np.sum(self.b,axis=2,keepdims=True)
            
        plt.plot(lp)
        plt.show()
    
    def predict(self, x):
        """
        alpha, c = self.forward(x)
        beta = self.backward(x)
        # (n,samples)<-(n,k,samples)
        px = np.sum(alpha[:,:,:,0]*beta[:,:,:,0],axis=1)
        
        return np.argmax(px,axis=0)
        """
        alpha, c = self.forward(x, scaling=True)
        # (n,samples)<-(n,samples,t)
        lpx = np.sum(np.log(c),axis=2)
        
        return np.argmax(lpx,axis=0)

    def sampling(self, samples, time):
        x = np.zeros((self.n,samples,time),dtype=np.int)
        z = np.zeros((self.n,samples,time),dtype=np.int)
        s = np.zeros((samples,self.k),dtype=np.int)
        idx = np.arange(samples,dtype=np.int)
        status = np.arange(self.k,dtype=np.int)
        outputs = np.arange(self.m,dtype=np.int)
        labels = np.zeros(samples,dtype=np.int)

        for i in range(self.n):
            labels = np.random.choice(status,size=samples,p=self.pi[i,:])
            for j in range(self.k):
                s[:,j] = np.random.choice(outputs,size=samples,p=self.b[i,j])
            x[i,idx,0] = s[idx,labels]
            z[i,:,0] = labels

        for t in range(1,time):
            for i in range(self.n):
                for j in range(self.k):
                    s[:,j] = np.random.choice(status,size=samples,p=self.a[i,j])
                labels = s[idx,labels]
                for j in range(self.k):
                    s[:,j] = np.random.choice(outputs,size=samples,p=self.b[i,j])
                x[i,idx,t] = s[idx,labels]
                z[i,:,t] = labels
        
        return x, z

    def log_plaus(self, x, z):
        # (n,samples)
        pini = self.pi[:,z[:,0]]
        # (n,samples,t-1)
        ptrans = self.a[:,z[:,:-1],z[:,1:]]
        # (n,samples,t)
        pout = self.b[:,z,x]
        
        lp = np.log(pini)
        lp += np.sum(np.log(ptrans),axis=2)
        lp += np.sum(np.log(pout),axis=2)

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

    print(np.eye(b.shape[2])[:,output].shape)
    
    model = Hmm(pi=pi[:,:,0],a=a,b=b)
    pred = model.predict(output)

    labels = np.arange(model.n)
    cm = confusion_matrix(answer,pred,labels=labels)
    print(cm)
    
    model = Hmm(pi=pi[0:1,:,0],a=a[0:1],b=b[0:1])
    x,z = model.sampling(100,50)
    lpxz,zpred = model.viterbi(x[0])
    lpz = model.log_plaus(x[0],z[0])
    lpzpred= model.log_plaus(x[0],zpred[0])
    # print(lpz-lpzpred)
    # labels = np.arange(model.k)
    # cm = confusion_matrix(z.ravel(),zpred.ravel(),labels=labels)
    # print(cm)

    model = Hmm(pi=pi[:,:,0],a=a,b=b)
    model.fit(output,answer)

    # print(a-model.a)

if __name__=="__main__":
    main()