import numpy as np 
import math 

def mid_price(lowest_ask_price, highest_bid_price,):
    Pa = lowest_ask_price
    Pb = highest_bid_price
    
    M = 0.5 * (Pa + Pb)
    return M

def imbalance(ask_price_size, bid_price_size):
    
    Qa = ask_price_size #total volume at the best ask
    Qb = bid_price_size #total volume at the best bid

    I = Qb / (Qa + Qb)
    return I 

def weighted_mid_price(lowest_ask_price, highest_bid_price, imbalance):
    
    Pa = lowest_ask_price
    Pb = highest_bid_price
    I = imbalance
    
    W = I * Pa + (1 - I) * Pb
    return W

def G():
    pass

def bid_ask_spread(ask_price_size, bid_price_size): 
    Pa = ask_price_size
    Pb = bid_price_size

    S = Pa - Pb
    return S

def micro_price(mid_price, imbalance, bid_ask_spread):
    M = mid_price
    I = imbalance
    S = bid_ask_spread

    P = M + G(I, S)
    return P 



if __name__ == "__main__":
    print("hello world")