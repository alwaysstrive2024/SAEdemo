import requests

# 比如你想获取 gemma-2-2b 的某个特征的真实社区标签
model_id = ["gemma-2-2b",'gpt2-small']
sae_id = ['20-gemmascope-res-16k',"0-res-jb"]
feature_id = [10432,10650]  # 举例一个特征ID

headers = {
    'x-api-key': 'sk-np-zFNEKNq3NLTEHxuxOyju47xq1WAbwzcuYBTJPBBPYlg0'
}
# for idx, i in enumerate(model_id):
#     url = f"https://www.neuronpedia.org/api/feature/{i}/{sae_id[idx]}/{feature_id[idx]}"
#     response = requests.get(url,headers=headers).json()
#     status_code = response.get("status_code")
#     if status_code != 200:
#         print(f"Error fetching feature #{feature_id[idx]} for model '{i}': {response.get('detail', 'Unknown error')}")
#         continue
#     print("Status Code:", response.get("status_code", "No status code"))

#     print(f"Feature #{feature_id[idx]} 的信息: {response}")

#     concept_label = response.get("explanation", "未找到标签")
#     print(f"Feature #{feature_id[idx]} 的真实标签是: {concept_label}")
current_model_id = model_id[0]
current_sae_id = sae_id[0]
current_feature_id = feature_id[0]

url = f"https://www.neuronpedia.org/api/feature/{current_model_id}/{current_sae_id}/{current_feature_id}"
response = requests.get(url,headers=headers).json()
status_code = response.get("status_code")
if status_code != 200:
    print(f"Error fetching feature #{current_feature_id} for model '{current_model_id}': {response.get('detail', 'Unknown error')}")
    
print("Status Code:", response.get("status_code", "No status code"))

print(f"Feature #{current_feature_id} 的信息: {response}")

concept_label = response.get("explanation", "未找到标签")
print(f"Feature #{current_feature_id} 的真实标签是: {concept_label}")