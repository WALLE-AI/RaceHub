import loguru
import pandas as pd

data_example = {
  "data": [
    {
      "callid": "gt_index123",
      "callid_no": "1",
      "label": "信用卡办理",
      "record_dt_info": "您请问是王新红王王先生吗您这样的吗",
      "对话角色": "CLIENT"
    },
    {
      "callid": "gt_index123",
      "callid_no": "",
      "label": "信用卡办理",
      "record_dt_info": "好的，请后续会转达给您的，请问您还过还有其他的事情吗？",
      "对话角色": "AGENT"
    },
    {
      "callid": "gt_index123",
      "callid_no": "2",
      "label": "信用卡办理",
      "record_dt_info": "这边广发银行信用卡中心的",
      "对话角色": "CLIENT"
    },
    {
      "callid": "gt_index123",
      "callid_no": "3",
      "label": "信用卡办理",
      "record_dt_info": "我们这边是广发银行信用卡中心的",
      "对话角色": "CLIENT"
    },
    {
      "callid": "gt_index123",
      "callid_no": "",
      "label": "信用卡办理",
      "record_dt_info": "请问您说的是哪家银行的信用卡？",
      "对话角色": "AGENT"
    },
    {
      "callid": "gt_index123",
      "callid_no": "4",
      "label": "信用卡办理",
      "record_dt_info": "就是说现在地有",
      "对话角色": "CLIENT"
    },
    {
      "callid": "gt_index123",
      "callid_no": "",
      "label": "信用卡办理",
      "record_dt_info": "能申请到的额度就是多少？",
      "对话角色": "AGENT"
    },
    {
      "callid": "gt_index123",
      "callid_no": "5",
      "label": "信用卡办理",
      "record_dt_info": "六万元",
      "对话角色": "CLIENT"
    },
    {
      "callid": "gt_index123",
      "callid_no": "",
      "label": "信用卡办理",
      "record_dt_info": "最近办卡有什么优惠活动吗？",
      "对话角色": "AGENT"
    },
    {
      "callid": "gt_index123",
      "callid_no": "6",
      "label": "信用卡办理",
      "record_dt_info": "就后三十六期",
      "对话角色": "CLIENT"
    },
    {
      "callid": "gt_index123",
      "callid_no": "",
      "label": "信用卡办理",
      "record_dt_info": "好的，把刚才说的都记下来了，后续有意向会再联系您，再见！",
      "对话角色": "AGENT"
    }
  ]
}

def data_example_process():
    data_dict_list = []
    for data  in data_example["data"]:
        data_dict = {}
        data_dict["callid"] = data['callid']
        data_dict["callid_no"] = data['callid_no']
        data_dict["label"] = data['label']
        data_dict["record_dt_info"] = data['record_dt_info']
        data_dict['对话角色']=data['对话角色']
        data_dict_list.append(data_dict)
    pd_data = pd.DataFrame(data_dict_list)
    pd_data.to_csv("data/data_example.csv",index=False)
    
def read_dataset_file(data_file_path):
    '''
    需要抽取client字段中 所有record_dt_info以及对应label 
    '''
    data = pd.read_csv(data_file_path)
    return data
    
    
    
    
if __name__ == "__main__":
    loguru.logger.info("race hub data preprocss start")
    data_example_process()
    

        