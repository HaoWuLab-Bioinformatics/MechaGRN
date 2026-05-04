import gzip
import io
import requests
import  pandas as pd
import numpy as np
import json
import jsonpath
import urllib.request, urllib.parse

def Judge_regulatory(TF, Target):
    try:
        url = "http://guolab.wchscu.cn/hTFtarget/api/quick_search?"
        data = {
            'kw' : Target
        }
        headers = {
        'referer':'https://guolab.wchscu.cn/hTFtarget/',
        "cookie": "acw_tc=781bad4417767608678462454e41fe7525dc5edf9640b1f3bf5d334e913044; acw_sc__v2=69e73823e4091e8791b355473c816bd868263a7f; ssxmod_itna=1-Qqmx0DBD27lQG0WGCDwx3qWw9Cx9DxBP01HD_xQ5DODLxn4xGdq23DyAKzkM2vnwYq_iDAr5D/WmoeDZDGIdDqx0EiUY_0GUzKRAWrb49IAi5rk9t_7DBxYiCjvvwKAvlQlcGaCtZ7_9EaODCPDExGkPAqxhDiiTx0rD0eDPxDYDGRWD7PDoxDr19YDj7ovG3fpizxDKx0kDY5KF4GWDiPD7OorKBhxF8khO8DDBO02KiiFqDi3F/Oh0zrPDiyDsfG4F4G1AD0HOBZXDDyyhkiEsz19QsemUS3DvxDkH7Ov5FrQOz8A=RCY30rWmFWFYRib7q4IDXKG5Q4q1A=YbKlbYeu=Bpxtw5nrYen5qxD83mGmY3AwXTzcMzgdKh7YWQix3G9GD/3qzDxwg_SD=Rox9CvdoqT74NnIOGrQB_TYe4D; ssxmod_itna2=1-Qqmx0DBD27lQG0WGCDwx3qWw9Cx9DxBP01HD_xQ5DODLxn4xGdq23DyAKzkM2vnwYq_iDAoeGI4OYmbI377kGUIdoAuKNsYjr7kKfG=YD",
         "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
        }

        response = requests.get(url=url, params=data, headers=headers)
        content = response.text

        with open('HTF.json', mode='w', encoding='utf-8') as f:
            f.write(content)

        obj = json.load(open('HTF.json', mode='r', encoding='utf-8'))
        Query_gene = jsonpath.jsonpath(obj, '$..target')
        Query_gene_information = Query_gene[0][0]['id']

        second_url = "http://guolab.wchscu.cn/hTFtarget/api/chipseq/targets/target?"
        data_second = {
            'target':Query_gene_information
        }
        headers_second = {
            "cookie": "acw_tc=781bad4417767608678462454e41fe7525dc5edf9640b1f3bf5d334e913044; acw_sc__v2=69e73823e4091e8791b355473c816bd868263a7f; ssxmod_itna=1-Qqmx0DBD27lQG0WGCDwx3qWw9Cx9DxBP01HD_xQ5DODLxn4xGdq23DyAKzkM2vnwYq_iDAr5D/WmoeDZDGIdDqx0EiUY_0GUzKRAWrb49IAi5rk9t_7DBxYiCjvvwKAvlQlcGaCtZ7_9EaODCPDExGkPAqxhDiiTx0rD0eDPxDYDGRWD7PDoxDr19YDj7ovG3fpizxDKx0kDY5KF4GWDiPD7OorKBhxF8khO8DDBO02KiiFqDi3F/Oh0zrPDiyDsfG4F4G1AD0HOBZXDDyyhkiEsz19QsemUS3DvxDkH7Ov5FrQOz8A=RCY30rWmFWFYRib7q4IDXKG5Q4q1A=YbKlbYeu=Bpxtw5nrYen5qxD83mGmY3AwXTzcMzgdKh7YWQix3G9GD/3qzDxwg_SD=Rox9CvdoqT74NnIOGrQB_TYe4D; ssxmod_itna2=1-Qqmx0DBD27lQG0WGCDwx3qWw9Cx9DxBP01HD_xQ5DODLxn4xGdq23DyAKzkM2vnwYq_iDAoeGI4OYmbI377kGUIdoAuKNsYjr7kKfG=YD",
            "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
        }

        data_second = urllib.parse.urlencode(data_second)
        second_url = second_url + data_second
        Request = urllib.request.Request(url=second_url, headers=headers_second)
        response = urllib.request.urlopen(Request)
        content = response.read()
        if content[:2] == b'\x1f\x8b':
            print("检测到 GZIP 压缩，正在解压...")
            try:
                # 使用 gzip 解压
                with gzip.GzipFile(fileobj=io.BytesIO(content)) as f:
                    content = f.read().decode('utf-8')  # 解压后通常是 utf-8
            except Exception as e:
                # 如果解压失败，尝试 gbk
                content = content.decode('gbk', errors='ignore')
        else:
            # 如果不是 gzip，尝试直接解码 (先试 utf-8，再试 gbk)
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                content = content.decode('gbk', errors='ignore')

        with open('HTF_list.json', mode='w', encoding='utf-8') as f:
            f.write(content)

        obj = json.load(open('HTF_list.json', mode='r', encoding='utf-8'))
        TF_list = jsonpath.jsonpath(obj, '$..tf_id')
        result = TF in TF_list
    except IndexError as e:
        return -1
    else:
        return result

if __name__ == '__main__':
    cell_type = 'mHSC-GM'
    TF = 'HCFC1'
    data = pd.read_csv(f'./Regulatory_relationship/{cell_type}_{TF}.csv', index_col=0)
    TF = data.iloc[0, 0]
    Targets = data.iloc[:, 1]

    result = []
    for target in Targets:
        relationship = Judge_regulatory(TF, target)
        result.append(relationship)
        print(f"{target} 受 {TF} 的调控" if  relationship else f"{target} 不受 {TF} 的调控")

    data['relation'] = np.array(result)
    data.to_csv(f"./Regulatory_relationship/{cell_type}_{TF}_relationship.csv")




