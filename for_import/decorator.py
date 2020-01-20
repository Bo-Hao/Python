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

def report_success(func):
    import time 
    import smtplib
    from email.mime.text import MIMEText
    import pickle
    def wrapper(*args, **kwargs):
        t = time.time()
        ans = func(*args, **kwargs)
        t = time.time() - t
        print("function: ", func.__name__, " time cost:", t, "(sec)")
        
        with open("/Users/pengbohao/login_info.pickle", 'rb') as f:
            login_info = pickle.load(f)

        gmail_user = login_info[0] # user
        gmail_password = login_info[1] # password

        msg = MIMEText("function: "+ func.__name__ + '. Mission Complete! cost: ' + str(t) + ".(sec)")
        msg['Subject'] = 'Inform' # title
        msg['From'] = gmail_user # send
        msg['To'] = login_info[2] # receive

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465) # google recommend
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.send_message(msg)
        server.quit()

        print('Email sent!')

        return ans
    return wrapper
