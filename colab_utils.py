# -*- coding: utf-8 -*-
"""
Colab utility functions
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 100

def get_header(inmat):
  kS1,kS2=inmat.shape
  ls_header=[]
  tmprange=np.arange(kS2)
  for i in range(kS2):
    ls_header.append(str(tmprange[i]+1))
  return ls_header

def np2dict(inmat):
    dict_mat=dict()
    dict_mat['headers']=get_header(inmat)
    dict_mat['data']=inmat.tolist()
    dict_mat['metadata']= None
    return dict_mat

def get_accuracy(alpha_cur,mu_cur,cova_cur,Xlab,ylab,kNum_lab_labels):
    #kNum_lab_labels
    kKh,kDh=mu_cur.shape
    ydum=np.arange(kNum_lab_labels)
    Xdum=np.random.rand(kKh,kDh)
    gnb_dum=GaussianNB()
    gnb_dum.fit(Xdum,ydum)
    #
    gnb_dum.class_prior_[:]=alpha_cur[:,0]
    gnb_dum.theta_[:,:]=mu_cur[:,:]
    gnb_dum.var_[:,:]=cova_cur[:,:]
    #
    inAcc= accuracy_score(gnb_dum.predict(Xlab),ylab-1)
    return inAcc,gnb_dum

def plot_gnb_semisup(X,y,gnb,inAcc,inTitle='Gaussian Naive Bayes Accuracy:',cmap='Paired_r'):
    colors=['white','blue','red','green','cyan','black','magenta','brown','purple','yellow']
    xx,yy,Z=form_grid(X,gnb)
    
    fig=plt.figure()
    #
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    #
    #plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap, edgecolors='k');
    labnos=np.unique(y)
    kLen=labnos.shape[0]
    for k in range(kLen):
        indk=(y==labnos[k])
        plt.scatter(X[indk,0], X[indk,1], c=colors[k], edgecolors='k');
    #plt.yticks(range(0,11))
    plt.xticks(fontsize=14)
    #plt.yticks(range(0,11))
    plt.yticks(fontsize=14)
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.grid()
    #
    strTitle=inTitle+', Accuracy on Labeled Data: %f' % (inAcc)
    #strTitle="Maximum likelihood Gaussian naive bayes accuracy: %f" % (inAcc)
    plt.title(strTitle,fontsize=20)
    #
    #Draw some axis aligned ellipsoids to give a rough idea
    linecols=['black','blue','green','cyan','brown','magenta','purple','red']
    m=0
    #
    if hasattr(gnb,'theta_'):
        #arElpCoefs=ls_arElpCoefs[m]
        cova=gnb.var_
        mu=gnb.theta_
        kK,kD=mu.shape
        for k in range(kK):
            covak=np.eye(2)
            covak[0,0]=cova[k,0]
            covak[1,1]=cova[k,1]
            #covak=covak/arElpCoef[k]
            val, rot = np.linalg.eig(covak)
            val = np.sqrt(val)
            center=np.zeros((2,1))
            center[:,0] = mu[k,0:2]
          
            t = np.linspace(0, 2.0 * np.pi, 1000)
            xy = np.stack((np.cos(t), np.sin(t)), axis=-1)
          
            #plt.scatter(x, y)
            plt.plot(*(rot @ (val * xy).T + center),color=linecols[m])
    #          
    #
    #
    plt.axis('scaled')
    return fig

def form_grid(X,mlmodel):
    h = 0.02
    #x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    x_min, x_max = X[:,0].min() - 100*h, X[:,0].max() + 100*h
    #y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    y_min, y_max = X[:,1].min() - 100*h, X[:,1].max() + 100*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = mlmodel.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return xx,yy,Z



def plot_dgnb_semisup(alpha_cur,mu_cur,cova_cur,X,ynew,Xlab,ylab,kNum_lab_labels,it):
    #Plot the result
    fig_cur=plt.figure()
    #
    inAcc,gnb_dum=get_accuracy(alpha_cur,mu_cur,cova_cur,Xlab,ylab,kNum_lab_labels)
    #
    inTitle='Total EM Iters: %d' % (it)
    #fig_cur=plot_gnb(X,ynew,gnb_dum,inAcc,inTitle)
    fig_cur=plot_gnb_semisup(X,ynew,gnb_dum,inAcc,inTitle)
    return fig_cur

