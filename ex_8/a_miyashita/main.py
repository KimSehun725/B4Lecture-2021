import argparse
import sys
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from matplotlib import pyplot as plt
import seaborn as sns


class Hmm:
    def __init__(self, k=None, m=None, pi=None, a=None, b=None):
        """
            Class for HMM.
            # Args
                k (int): number of status of latent variable
                m (int): number of status of output variable
                pi (ndarray, shape=(k,)): initial probability
                a (ndarray, shape=(k,k)): transition probability
                b (ndarray, shape=(k,m)): output probability
        """
        if pi is None:
            if k is None:
                sys.exit("Hmm.__init__:k is required")
            self.pi = np.zeros(k)
        else:
            self.pi = pi
        
        if a is None:
            if k is None:
                sys.exit("Hmm.__init__:k is required")
            self.a = np.zeros((k,k))
        else:
            self.a = a

        if b is None:
            if k is None:
                sys.exit("Hmm.__init__:k is required")
            if m is None:
                sys.exit("Hmm.__init__:m is required")
            self.b = np.zeros((k,m))
        else:
            self.b = b

        self.k = self.pi.shape[0]
        self.m = self.b.shape[1]

    def forward(self, x, scaling=False):
        """
            Calculate p(X^t,z_t) by forward algorithm.
            # Args
                x (ndarray, axis=(samples,t)): output labels
                scaling (bool, default=False): 
                    If scaling is True, this function return p(z_t|X^t) instead of p(X^t,z_t).
                    then the scaling coefficent is returned as c.
            # Returns
                alpha (ndarray, axis=(k,samples,t)): p(X^t,z_t) (or p(z_t|X^t))
                c (ndarray, axis=(samples,t)): scaling coefficent
        """
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
        """
            Calculate p(x_{t+1},...,x_{n}|z_t) by backward algorithm.
            # Args
                x (ndarray, axis=(samples,t)): output labels
                c (ndarray, axis=(samples,t)):
                    If c is given, output beta is scaled by c.
            # Returns
                beta (ndarray, axis=(k,samples,t)): (scaled) p(x_{t+1},...,x_{n}|z_t)
        """
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

    def viterbi(self, x, condition=False):
        """
            Calculate the most plausible latent variables by viterbi algorithm.
            # Args
                x (ndarray, axis=(samples,t)): output labels
                condition (bool, default=False):
                    If condition is True, it is assumed that all output labels are sampled 
                    from the distribution conditioned by same latent variables.
            # Returns
                lpmax (ndarray, axis=(samples,)): max log plausibility
                zmax (ndarray, axis=(samples,t)): the most plausible latent variables
        """
        samples = 1 if condition else x.shape[0]
        time = x.shape[1]
        
        # (k,samples,t)
        lp = np.log(self.b[:,x])
        if condition:
            lp = np.sum(lp,axis=1,keepdims=True)
        # (k,K',1)
        la = np.log(self.a[:,:,np.newaxis])
        # (k,samples,t)
        w = np.zeros((self.k,samples,time),dtype=np.int)
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
        """
            Fit parameters by EM algorithm.
            # Args
                x (ndarray, axis=(samples,t)): output labels
                y: dummy
        """
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

            # check
            plaus = np.sum(np.log(c))
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
            # (k,samples,t)<-(k,samples,t)*(k,samples,t)
            gamma = alpha*beta
            # (k,k',samples,t)<-(k,1,samples,t)*(1,k',samples,t)
            xi = alpha[:,np.newaxis,:,:-1]*beta[:,:,1:]
            # (k,k',samples,t)<-(k,k',samples,t)*(k,k',1,1)*(1,k',samples,t)
            xi *= self.a[:,:,np.newaxis,np.newaxis]*pout[:,:,1:]
            # (k,k',samples,t)<-(k,k',samples,t)/(1,1,samples,t)
            xi /= c[:,1:]

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
    
    def predict(self, x, scaling=False):
        """
            Calculate log(p(X)).
            # Args
                x (ndarray, axis=(samples,t)): output labels
                scaling (bool, default=False):
                    If scaling is True, scaling forward algorithm is used.
            # Returns 
                lpx (ndarray, axis=(samples,)): log(p(X))
        """
        if scaling:
            alpha, c = self.forward(x, scaling=True)
            # (samples)<-(samples,t)
            lpx = np.sum(np.log(c),axis=1)
        else:
            alpha, c = self.forward(x, scaling=False)
            # (samples)<-(k,samples)
            px = np.sum(alpha[:,:,-1],axis=0)
            lpx = np.log(px)
        
        return lpx

    def sampling(self, samples, time, condition=False):
        """
            Sample data from HMM.
            # Args 
                samples (int): sample size
                time (int): size of series
                condition (bool, default=False):
                    If condition is True, all output labels sample are sampled 
                    from the distribution conditioned by same latent labels.
            # Returns
                x (ndarray, axis=(samples,t)): output labels
                z (ndarray, axis=(samples,t)): latent labels
        """
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
        if condition:
            x[:,0] = s[:,labels[0]]
            z[:,0] = labels[0]
        else:
            x[idx,0] = s[idx,labels]
            z[:,0] = labels

        for t in range(1,time):
            for j in range(self.k):
                s[:,j] = np.random.choice(status,size=samples,p=self.a[j])
            labels = s[idx,labels]
            for j in range(self.k):
                s[:,j] = np.random.choice(outputs,size=samples,p=self.b[j])
            if condition:
                x[:,t] = s[:,labels[0]]
                z[:,t] = labels[0]
            else:
                x[idx,t] = s[idx,labels]
                z[:,t] = labels
        
        return x, z

    def log_plaus(self, x, z):
        """
            Calculate log(p(X,Z)).
            # Args
                x (ndarray, axis=(samples,t)): output labels
                z (ndarray, axis=(samples,t)): latent labels
            # Returns 
                lpx (ndarray, axis=(samples,)): log(p(X,Z))
        """
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
    parser = argparse.ArgumentParser(description="HMM classification or etc")
    parser.add_argument('sc', type=str, help="input filename followed by extension .pickle")
    parser.add_argument('-v', '--viterbi', type=int, nargs=2, help="sampling size for viterbi (samplesize,seriessize)")
    parser.add_argument('-t', '--train',  type=int, nargs=2, help="sampling size for fit (samplesize,seriessize)")
    args = parser.parse_args()

    print(f">>> data = pickle.load(open({args.sc}.pickle, 'rb'))")
    data = pickle.load(open(f"{args.sc}.pickle", 'rb'))
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

    # predict by forward
    lps = np.zeros((modelsize,samplesize))
    for i in range(modelsize):
        lps[i] = models[i].predict(output, scaling=True)
    
    pred = np.argmax(lps,axis=0)
    labels = np.arange(modelsize)
    cm1 = confusion_matrix(answer,pred,labels=labels)

    # predict by viterbi
    lps = np.zeros((modelsize,samplesize))
    for i in range(modelsize):
        lps[i],_ = models[i].viterbi(output)
    
    pred = np.argmax(lps,axis=0)
    labels = np.arange(modelsize)
    cm2 = confusion_matrix(answer,pred,labels=labels)

    # plot
    fig, ax = plt.subplots(1, 2)
    fig.subplots_adjust(wspace=0.5)
    
    sns.heatmap(cm1, annot=True, cmap='Greys', ax=ax[0])
    ax[0].set(title=f"{args.sc} Forward", xlabel="Predicted model", ylabel="Answer model")
    sns.heatmap(cm2, annot=True, cmap='Greys', ax=ax[1])
    ax[1].set(title=f"{args.sc} Viterbi", xlabel="Predicted model", ylabel="Answer model")
    plt.show()
    
    # calculate the most plausible latent labels
    if args.viterbi:
        model = Hmm(pi=pi[0,:,0],a=a[0],b=b[0])
        x,z = model.sampling(args.viterbi[0],args.viterbi[1],condition=True)
        lpxzpred,zpred = model.viterbi(x,condition=True)
            
        labels = np.arange(model.k)
        cm = confusion_matrix(z[0].ravel(),zpred.ravel(),labels=labels)
        sns.heatmap(cm, annot=True, cmap='Greys')
        plt.title(f"{args.sc} viterbi")
        plt.xlabel("Predicted status")
        plt.ylabel("Answer status")
        plt.show()
        """
        tries = 100
        acc = np.zeros((args.viterbi[0]-1,tries))
        for i in range(1, args.viterbi[0]):
            for j in range(tries):
                x,z = model.sampling(i,args.viterbi[1],condition=True)
                lpxzpred,zpred = model.viterbi(x,condition=True)
                acc[i-1,j] = accuracy_score(z[0].ravel(),zpred.ravel())
        plt.plot(np.mean(acc,axis=1))
        plt.title("viterbi accuracy")
        plt.xlabel("sample size")
        plt.ylabel("accuracy")
        plt.show()
        """
    
    # train models
    if args.train:
        models_train = []
        for i in range(modelsize):
            models_train.append(Hmm(pi=pi[i,:,0],a=a[i],b=b[i]))

        start = time.time()
        for i in range(modelsize):
            train,_ = models[i].sampling(args.train[0],args.train[1])
            models_train[i].fit(train,ltr=False)
        print(time.time()-start)

        lps = np.zeros((modelsize,samplesize))
        for i in range(modelsize):
            lps[i] = models_train[i].predict(output, scaling=True)
        
        pred = np.argmax(lps,axis=0)
        labels = np.arange(modelsize)
        cm = confusion_matrix(answer,pred,labels=labels)
        sns.heatmap(cm, annot=True, cmap='Greys')
        plt.title(f"{args.sc} trained model")
        plt.xlabel("Predicted label")
        plt.ylabel("Answer label")
        plt.show()
    

if __name__=="__main__":
    main()