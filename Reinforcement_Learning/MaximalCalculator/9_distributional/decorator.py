##########################################
# 
# 
# 
##########################################

def timecost(func):
    import time
    def wrapper(*args, **kwargs):
        t = time.time()
        ans = func(*args, **kwargs)
        t = time.time() - t
        print("function: ", func.__name__, " time cost:", t, "(sec)")
        return ans
    return wrapper