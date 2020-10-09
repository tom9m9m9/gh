# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:25:38 2020

@author: tom9m
"""

import scipy.special
from sklearn import preprocessing
import numpy as np
import sys
import os
import pickle


def main ():

    digamma=scipy.special.digamma
    print('Please, input document data.')
    file1=input()
    print('Please, input vocabulary data.')
    file2=input()
    print('Please, the name of output')
    outname=input()
    with open(file1,"rb")as f0:
        documents_stem_id=pickle.load(f0)
    with open(file2,"rb")as f1:
        data=f1.readlines()
        V=len(data)
    print('Please,input the number of topics')
    K=int(input()) #トピック数の指定
    print('Please,input the number of epoch')
    epoch=int(input()) #トピック数の指定

    D=len(documents_stem_id)#文書数の指定
    #V=int(sys.argv[3])#語彙数の指定
    N_dk = np.zeros([D,K]) #文書dでトピックkが割り振られた単語数
    N_kv = np.zeros([K,V]) #文書集合全体で語彙vにトピックkが割り振られた単語数        
    N_k  = np.zeros([K,1]) #文書集合全体でトピックkが割り振られた単語数
    N_d=np.zeros([D,1])#各ドキュメントの長さ
    for d in range(D):
        N_d[d]=len(documents_stem_id[d])
        #文書dのn番目の単語に付与されたトピック
        theta_k_=np.zeros([K])
        phi_kv_=np.zeros([K,V])
    
    z_dn=[]
    for d in range(D):
        z_dn.append(np.random.randint(0, K,len(documents_stem_id[d])) )
        #N_dkとN_kについて
        for i in range(len(z_dn[d])):
            N_dk[d,z_dn[d][i]]+=1
            N_k[z_dn[d][i]]+=1
            #N_kvについて    
        for v,k in zip(documents_stem_id[d],z_dn[d]):
            N_kv[k,v]+=1
    
    alpha=np.ones([K],dtype='float')*50/K
    beta=0.1
    
    for i in range(epoch):
        print("Epoch: {}".format(i+1))
        numerator_p = 0
        denominator_p = 0
        loglikelihood=0
        for d in range(D):
            sys.stdout.write("\r%d / %d" % (d+1, D))
            sys.stdout.flush()
            for n in np.random.permutation(len(documents_stem_id[d])):#単語をバラバラに見る
                current_topic = z_dn[d][n]
                v=documents_stem_id[d][n]
                #if(current_topic>0):#自身のカウントを引く
                N_dk[d, current_topic] -= 1
                N_kv[current_topic, v] -= 1
                N_k[current_topic] -= 1
                theta_phi=0
                if (N_kv[current_topic, v]<0):
                    print(N_kv[current_topic, v])
            
                    
                    #サンプリング確率と尤度を計算-----------------------------------------------------------
                p_z_dn = np.zeros(K)
                theta_phi=0
                for k in range(K):
                            
                    A = N_dk[d,k] + alpha[k]
                    B = (N_kv[k,v] + beta)/(N_k[k] + beta*V)
                            
                    p = A * B 
                    if(p  < 0):
                        break
                    p_z_dn[k] = p
                            
                    theta_k = (N_dk[d,k]+alpha[k]) / (N_d[d]+alpha[k]*K) # 
                           
                    theta_k_[k]=theta_k[0]
                    
                    phi_kv = (N_kv[k,v]+beta) /(N_k[k]+beta*V) #
                    phi_kv_[k,v]=phi_kv[0]
                    theta_phi +=theta_k*phi_kv
                            
                    loglikelihood += np.log(theta_phi)
                    p_z_dn = preprocessing.normalize(p_z_dn.reshape(1, -1), norm="l1")[0] # 正規化
            
                    #-------------------------------------------------------------------------------
                        
                    #カテゴリカル分布を使って文書dのn番目の単語のトピックをサンプリング   
                    new_topic=np.argmax(np.random.multinomial(1, p_z_dn, size=1))#最大となるインデックスを返す
                    z_dn[d][n]=new_topic
                                      
                    N_dk[d, new_topic] += 1
                    N_kv[new_topic, v] += 1
                    N_k[new_topic] += 1
                numerator_p += loglikelihood
                denominator_p += N_d[d]
            
            
                        #  パラメータ更新
                        #α トピック分布用のパラメータ
        for k in range(K):
            numerator=0
            denominator=0
            for d in range(D):
                numerator +=digamma(N_dk[d][k]+alpha[k])- digamma(alpha[k])
                denominator += digamma(N_d[d]+np.sum(alpha))- digamma(np.sum(alpha))
            alpha[k] = alpha[k]*(numerator / denominator)
            if(alpha[k]<=0):
                alpha[k]=0.1
        print('alpha:{}'.format(alpha))
    
                        
                        #β 単語分布用のパラメータ
        numerator = np.sum(digamma(N_kv+beta)) - K*V*digamma(beta)
        denominator = V*(np.sum(digamma(N_k+beta*V)) - K*digamma(beta*V))
        beta = beta*(numerator / denominator)
            
    
                    #パラメータ出力
        print("\nparameters")
        print("alpha :{}".format(alpha))
        print("beta :{}".format(beta))
    if not os.path.isdir('n_kv'):
        os.makedirs('n_kv')
    if not os.path.isdir('n_k'):
        os.makedirs('n_k')
    if not os.path.isdir('n_dk'):
        os.makedirs('n_dk')   
    if not os.path.isdir('theta_k'):
        os.makedirs('theta_k')  
    if not os.path.isdir('phi_kv'):
        os.makedirs('phi_kv')           
    np.savetxt('n_kv/n_kv_{}.txt'.format(outname),N_kv)   
    np.savetxt('n_k/n_k_{}.txt'.format(outname),N_k)
    np.savetxt('n_dk/n_dk_{}.txt'.format(outname),N_dk)
    np.savetxt('theta_k/theta_k_{}.txt'.format(outname),theta_k_)
    np.savetxt('phi_kv/phi_kv_{}.txt'.format(outname),phi_kv_)
        
        
if __name__ == "__main__":
    main ()   
