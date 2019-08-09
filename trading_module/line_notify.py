def LineNotify(Message, LineToken):
    import requests
    url  = "https://notify-api.line.me/api/notify"
    data = ({'message':Message})
    LINE_HEADERS = {"Authorization":"Bearer " + LineToken}
    session  = requests.Session()
    response = session.post(url, headers=LINE_HEADERS, data=data)
    return response

def LineNotifyImage(Message, ImageFile, LineToken):
    import requests
    url  = "https://notify-api.line.me/api/notify"
    data = ({'message': Message})
    file = {'imageFile': open(ImageFile, 'rb')}
    LINE_HEADERS = {"Authorization":"Bearer " + LineToken}
    session  = requests.Session()
    response = session.post(url, headers=LINE_HEADERS, files=file, data=data)
    return response

