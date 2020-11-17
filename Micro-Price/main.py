import pickle 
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.linalg import block_diag


def prep_data_sym(T,n_imb,dt,n_spread,hist):
    data.rename(columns={'bid1': 'bid'}, inplace=True)
    data.rename(columns={'ask1': 'ask'}, inplace=True)
    data.rename(columns={'askVol1': 'as'}, inplace=True)
    data.rename(columns={'bidVol1': 'bs'}, inplace=True)
    data.rename(columns={'LocalTime': 'time'}, inplace=True)    
    #Removed data(last param) from the original function
    spread=T.ask-T.bid
    ticksize=np.round(min(spread.loc[spread>0])*1e6)/1e6
    T.spread=T.ask-T.bid
    # adds the spread and mid prices
    T['spread']=np.round((T['ask']-T['bid'])/ticksize)*ticksize
    T['mid']=(T['bid']+T['ask'])/2

    #filter out spreads >= n_spread
    T = T.loc[(T.spread <= n_spread*ticksize) & (T.spread>0)]
    T['imb']=T['bs']/(T['bs']+T['as'])
    T['wmid']=T['imb']*T['ask']+(1-T['imb'])*T['bid']

    #discretize imbalance into percentiles
    T['imb_bucket'] = pd.qcut(T['imb'], n_imb, labels=False)
    T['next_mid']=T['mid'].shift(-dt)
    T['next_wmid']=T['wmid'].shift(-dt)

    #step ahead state variables
    T['next_spread']=T['spread'].shift(-dt)
    T['next_time']=T['time'].shift(-dt)
    T['next_imb_bucket']=T['imb_bucket'].shift(-dt)

    # step ahead change in price
    T['dM']=np.round((T['next_mid']-T['mid'])/ticksize*2)*ticksize/2
    T['dW']=T['next_wmid']-T['wmid']
    T = T.loc[(T.dM <= ticksize*1.1) & (T.dM>=-ticksize*1.1)]

    #Getting the next step done
    T2 = T.copy(deep=True)
    T2['imb_bucket']=n_imb-1-T2['imb_bucket']
    T2['next_imb_bucket']=n_imb-1-T2['next_imb_bucket']
    T2['dM']=-T2['dM']
    T2['dW']=-T2['dW']
    T2['mid']=-T2['mid']
    T2['wmid']=-T2['wmid']
    T=pd.concat([T,T2])
    T.index = pd.RangeIndex(len(T.index))
    
    return T,ticksize

def estimate_old(T):
    no_move=T[T['dM']==0]
    no_move_counts=no_move.pivot_table(index=[ 'next_imb_bucket'], 
                     columns=['spread', 'imb_bucket'], 
                     values='time',
                     fill_value=0, 
                     aggfunc='count').unstack()

    #print no_move_counts
    Q_counts=np.resize(np.array(no_move_counts[0:(n_imb*n_imb)]),(n_imb,n_imb))
    # loop over all spreads and add block matrices
    for i in range(1,n_spread):
        Qi=np.resize(np.array(no_move_counts[(i*n_imb*n_imb):(i+1)*(n_imb*n_imb)]),(n_imb,n_imb))
        Q_counts=block_diag(Q_counts,Qi)
    #print Q_counts
    move_counts=T[(T['dM']!=0)].pivot_table(index=['dM'], 
                         columns=['spread', 'imb_bucket'], 
                         values='time',
                         fill_value=0, 
                         aggfunc='count').unstack()

    R_counts=np.resize(np.array(move_counts),(n_imb*n_spread,4))
    T1=np.concatenate((Q_counts,R_counts),axis=1).astype(float)
    for i in range(0,n_imb*n_spread):
        T1[i]=T1[i]/T1[i].sum()
    Q=T1[:,0:(n_imb*n_spread)]
    R1=T1[:,(n_imb*n_spread):]

    K=np.array([-0.01, -0.005, 0.005, 0.01])
    move_counts=T[(T['dM']!=0)].pivot_table(index=['spread','imb_bucket'], 
                     columns=['next_spread', 'next_imb_bucket'], 
                     values='time',
                     fill_value=0, 
                     aggfunc='count') #.unstack()

    R2_counts=np.resize(np.array(move_counts),(n_imb*n_spread,n_imb*n_spread))
    T2=np.concatenate((Q_counts,R2_counts),axis=1).astype(float)

    for i in range(0,n_imb*n_spread):
        T2[i]=T2[i]/T2[i].sum()
    R2=T2[:,(n_imb*n_spread):]
    Q2=T2[:,0:(n_imb*n_spread)]
    G1=np.dot(np.dot(np.linalg.inv(np.eye(n_imb*n_spread)-Q),R1),K)
    B=np.dot(np.linalg.inv(np.eye(n_imb*n_spread)-Q),R2)
    
    return G1,B,Q,Q2,R1,R2,K

def compute_next_price_moves(T):
    T['npm']=np.nan
    T['npm'].loc[(T['dM'] != 0 )]=np.round(T['dM'].loc[(T['dM'] != 0 )] /ticksize*2)*ticksize/2
    T['npm']=T['npm'].bfill()
    T = T.loc[(T.npm <= ticksize*1.1)& (T.npm>=-ticksize*1.1) ]
    T['nnpm']=np.nan
    T['nnpm'].loc[(T['dM'] != 0 )]=T['dM'].loc[(T['dM'] != 0 )]+T['npm'].shift(-1).loc[(T['dM'] != 0 )]
    T['nnpm']=T['nnpm'].bfill()
    T = T.loc[(T.nnpm <= ticksize*2.1)& (T.nnpm>=-ticksize*2.1) ]
    T['nnnpm']=np.nan
    T['nnnpm'].loc[(T['dM'] != 0 )]=T['dM'].loc[(T['dM'] != 0 )]+T['nnpm'].shift(-1).loc[(T['dM'] != 0 )]
    T['nnnpm']=T['nnnpm'].bfill()
    T = T.loc[(T.nnnpm <= ticksize*3.1)& (T.nnnpm>=-ticksize*3.1) ]
    T['nnnnpm']=np.nan
    T['nnnnpm'].loc[(T['dM'] != 0 )]=T['dM'].loc[(T['dM'] != 0 )]+T['nnnpm'].shift(-1).loc[(T['dM'] != 0 )]
    T['nnnnpm']=T['nnnnpm'].bfill()
    T = T.loc[(T.nnnnpm <= ticksize*4.1)& (T.nnnnpm>=-ticksize*4.1) ]
    T['nnnnnpm']=np.nan
    T['nnnnnpm'].loc[(T['dM'] != 0 )]=T['dM'].loc[(T['dM'] != 0 )]+T['nnnnpm'].shift(-1).loc[(T['dM'] != 0 )]
    T['nnnnnpm']=T['nnnnnpm'].bfill()
    T = T.loc[(T.nnnnpm <= ticksize*5.1)& (T.nnnnpm>=-ticksize*5.1) ]
    T['nnnnnnpm']=np.nan
    T['nnnnnnpm'].loc[(T['dM'] != 0 )]=T['dM'].loc[(T['dM'] != 0 )]+T['nnnnnpm'].shift(-1).loc[(T['dM'] != 0 )]
    T['nnnnnnpm']=T['nnnnnnpm'].bfill()
    T = T.loc[(T.nnnnpm <= ticksize*6.1)& (T.nnnnpm>=-ticksize*6.1) ]
    return T

if __name__ == "__main__":

    with open("/Users/pengbohao/Downloads/BTCorderbook.pickle", "rb") as f:
        data = pickle.load(f)

    # [_id, ask, ask1, ask2, ask3, bid, bid1, bid2, bid3, close, high, low, open, volume, width, Key, LastUpdate]

    data['LastUpdate'] = pd.to_datetime(data['LastUpdate'], format="%Y%m%d:%H:%M:%S.%f")
    data = data.sort_index(ascending=False)
    data.index = data["LastUpdate"]


    data['Spread'] = 100*(data['ask'] - data['bid'])/data['bid']
    data['MidPrice'] = (data['ask'] + data['bid'])/2
    data['CompositePrice'] = (data['ask'] * data['bid1'] + data['bid']*data['ask1'])/(data['ask1'] + data['bid1'])
    data['MidPriceReturn'] = data['MidPrice'].pct_change()


    data['LastUpdate'] = data['LastUpdate'].apply(lambda x : pd.to_datetime(x, unit= 'ms') )

    data = data.resample('1000L',closed = 'left', label='right').agg('last')
    data = data.ffill()

    print(data)

    print(100*sum(data['MidPriceReturn'] != 0)/len(data))
    print(100* sum( (data['ask1'] <= data['bid1'].shift(1)) | (data['bid1'] >= data['ask1'].shift(1)) ) / len(data))

    data.rename(columns={'ask1': 'as'}, inplace=True)
    data.rename(columns={'bid1': 'bs'}, inplace=True)
    data.rename(columns={'LastUpdate': 'time'}, inplace=True)    

    


    n_imb=7
    n_spread=2
    dt=1
    



    ticker='ETHBTC'
    T,ticksize=prep_data_sym(data,n_imb,dt,n_spread,'off')
    imb=np.linspace(0,1,n_imb)
    G1,B,Q,Q2,R1,R2,K=estimate_old(T)
    T=compute_next_price_moves(T)

    print(G1.shape, B.shape, R1.shape, R2.shape)



    """ T['dM']=T['mid'].shift(dt)-T['mid']
    T['dW']=T['wmid'].shift(dt)-T['mid']
    grouped=T.groupby(['spread','imb_bucket'])
    T['dt']=(T['EventTime'].shift(-dt)-T['EventTime'])

    grouped=T.groupby(['spread','imb_bucket'])
    res1min=grouped['dM'].aggregate(np.mean)
    std1=grouped['dM'].aggregate(np.std)
    cnt1=grouped.count()['dM']

    grouped2=T.groupby(['spread','imb_bucket'])
    res1min2=grouped2['dW'].aggregate(np.mean)
    std12=grouped2['dW'].aggregate(np.std)
    cnt12=grouped2.count()['dW']

    i =1
    plt.figure()
    #plt.scatter(imb,g_star[(0+i*n_imb):(n_imb+i*n_imb)],label="G_star",color='r',s=50)
    plt.errorbar(imb,res1min[(0+i*n_imb):(n_imb+i*n_imb)],yerr=np.sqrt(dt)*std1[(0+i*n_imb):(n_imb+i*n_imb)]/np.sqrt(cnt1[(0+i*n_imb):(n_imb+i*n_imb)]), capthick=5,label='empirical E[M_{t+'+str(dt)+'s}-M_t]')

    plt.errorbar(imb,res1min2[(0+i*n_imb):(n_imb+i*n_imb)],yerr=np.sqrt(dt)*std12[(0+i*n_imb):(n_imb+i*n_imb)]/np.sqrt(cnt12[(0+i*n_imb):(n_imb+i*n_imb)]), capthick=5,label='empirical E[M_{t+'+str(dt)+'s}-M_t]')
    plt.legend(loc='upper left')
    plt.title(ticker+' microprice vs. empirical,spread='+str(i+1))
    plt.xlabel('Imbalance')
    plt.title(ticker+' microprice vs. empirical,spread='+str(i+1)) """