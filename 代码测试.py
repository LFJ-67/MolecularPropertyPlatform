import requests

base = "http://127.0.0.1:5000"

# 1. 上传文件（确保 test.csv 在当前目录）
files = {'file': open('test.csv', 'rb')}
r = requests.post(f'{base}/upload', files=files)
print('Upload:', r.json())
if not r.json().get('success'):
    exit()

filename = r.json()['filename']

# 2. 发起批量预测
payload = {'filename': filename, 'target': 'triplet'}
r = requests.post(f'{base}/batch_predict', json=payload)
print('Batch:', r.json())
if not r.json().get('success'):
    exit()

download_name = r.json()['download']

# 3. 下载结果
r = requests.get(f'{base}/download/{download_name}')
if r.status_code == 200:
    with open(download_name, 'wb') as f:
        f.write(r.content)
    print(f'Downloaded {download_name}')
else:
    print('Download failed')