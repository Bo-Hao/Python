def report_success():
    import smtplib
    from email.mime.text import MIMEText
    import pickle 
    with open("/Users/pengbohao/login_info.pickle", 'rb') as f:
        login_info = pickle.load(f)
    gmail_user = login_info[0] # user
    gmail_password = login_info[1] # password

    msg = MIMEText('Mission Complete!')
    msg['Subject'] = 'Inform' # title
    msg['From'] = gmail_user # send
    msg['To'] = login_info[2] # receive

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465) # google recommend
    server.ehlo()
    server.login(gmail_user, gmail_password)
    server.send_message(msg)
    server.quit()

    print('Email sent!')

