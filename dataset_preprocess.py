import json
import loguru
import pandas as pd

from prompt import NER_TASK_LABEL_PROMPT

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
{"text": "彭久洋：我的魂飞了贝鲁斯科尼老古董收藏家（图）", "label": {"name": {"彭久洋": [[0, 2]], "贝鲁斯科尼": [[9, 13]]}, "position": {"收藏家": [[17, 19]]}}}
{"text": "会议批准了中国与欧盟海军、多国海上力量和北约等就在“国际推荐通行走廊”", "label": {"government": {"中国与欧盟海军": [[5, 11]], "北约": [[20, 21]]}}}
{"text": "他们需要1分确保小组出线。出线形势要求赫塔必须全力争胜。interwetten相同赔率下，", "label": {"organization": {"赫塔": [[19, 20]], "interwetten": [[28, 38]]}}}


ner_data_example = [
  {
    "text":"北京勘察设计协会副会长兼秘书长周荫如",
    "label":'''{"organization": {"北京勘察设计协会": [[0, 7]]}, "name": {"周荫如": [[15, 17]]}, "position": {"副会长": [[8, 10]], "秘书长": [[12, 14]]}}'''
  },
  {
    "text":"彭久洋：我的魂飞了贝鲁斯科尼老古董收藏家（图）",
    "label":'''{"name": {"彭久洋": [[0, 2]], "贝鲁斯科尼": [[9, 13]]}, "position": {"收藏家": [[17, 19]]}}'''
    
  },
  {
    "text":"会议批准了中国与欧盟海军、多国海上力量和北约等就在“国际推荐通行走廊",
    "label":'''{"government": {"中国与欧盟海军": [[5, 11]], "北约": [[20, 21]]}}'''
    
  },
  {
    "text":"他们需要1分确保小组出线。出线形势要求赫塔必须全力争胜。interwetten相同赔率下",
    "label":'''{"organization": {"赫塔": [[19, 20]], "interwetten": [[28, 38]]}}'''
    
  }

  ]

def ner_data_example_process():
    data_dict_list = []
    for data  in ner_data_example:
        data_dict = {}
        data_dict["text"] = data['text']
        data_dict["label"] = data['label']
        data_dict_list.append(data_dict)
    pd_data = pd.DataFrame(data_dict_list)
    pd_data.to_csv("data/data_ner_example.csv",index=False)


def data_example_process():
    '''
    ##https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health 文本匹配的数据
      ## https://github.com/jeffheaton/app_generative_ai/tree/main
        training_data = [{"instruction": "判断以下医疗文本的科室分类","input": "患者出现头痛、恶心、视物模糊等症状","output": "神经内科"},# 更多训练样本...
    "conversations":[{"from":"human","value":"问：左踝关节扭伤5年未痊愈。2009年左踝扭伤至今已5年多，走路没多久就会酸疼，肿胀，如果走的时间稍>微长点，接下来的几天脚会很不舒服，酸疼，有些肿胀，早上起来肿胀会消退些。不走路的时候踝关节里也会有不舒服的感觉。2009年1月因在不平坦的路上走路崴了脚，当时没有疼痛，于是没有进行治疗，不久后
        跑步就出现左踝关节肿胀疼痛，敷了几剂药和泡了点中药后消了一点肿，疼痛也不明显，以为没有大碍，因为当时还在上学就没有再做进一步治疗。但是后来发现走路之后会酸痛肿胀，晚上会比较明显，早上起床>就会消点肿，休息过后肿痛会好转，于是又继续敷药，但也没什么起色。2009年9月，大学军训了一个星期左右，前几天要长期站着，后来因为脚疼也申请了休息。大学期间也免不了要做些运动，但还是尽量避免，
      发觉踝关节很容易就累，就酸痛，不能走很长时间。后来也做过理疗，针灸，但是效果不明显，只是感觉舒服一些。后来因为要出国读书就停止了治疗，平时也不做运动，尽量避免长时间走路。2012年7月做了核磁
      共振，到医院做了理疗，敷了药，"},{"from":"gpt","value":"答：作为医生，我建议您尽快前往医院进行进一步的检查和治疗。根据描述，您的左踝关节扭伤已经持续了5年，而且症状似乎没有得到缓解。这可能
      是由于初始扭伤时没有及时进行治疗，导致伤程加长并引起了慢性炎症。\n\n您之前尝试过一些治疗方法，如敷药、泡中药、理疗和针灸，但效果不明显。这可能是因为病因尚未明确，治疗方法不够全面和专业。>因此，我建议您前往医院接受全面的检查，可能包括X光、核磁共振或其他相关检查，以了解关节结构是否受损。\n\n一旦明确了病因，您可以与医生一起制定治疗计划。治疗可能包括物理治疗、康复训练、药物治
      疗或手术等，具体根据您的病情来定。同时，您也需要按医生建议进行生活方式调整，如适当休息、避免过度活动或长时间站立、做适当的运动来增强踝关节的稳定性。\n\n最重要的是，不要忽视疼痛和不适感，>及时就医是为了防止病情进一步恶化并提高痊愈的机会。祝您早日康复！"}]
    '''
    ##微调与不微调两种作为 
    # 1、一种使用通过API+prompt的方式完成对应NLP任务（NER 文本匹配文本 summary问题等等） 
    # 2、通过微调+prompt+rag（搜索api）的方式解决精度问题
    test_system_prompt = '''
    你是一个智能助手，你能够根据用户输入判断出文本的意图
    用户输入：{text}
    按如下json格式输出
    {
      "intent":""
    }
    '''
    system_prompt = "你是一个智能助手，能够高质量判断出文本的对话意图"
    data_dict_list = []
    for data  in data_example["data"]:
        data_dict = {}
        data_dict["callid"] = data['callid']
        data_dict["callid_no"] = data['callid_no']
        data_dict["label"] = data['label']
        data_dict["record_dt_info"] = data['record_dt_info']
        data_dict['对话角色']=data['对话角色']
        if data["对话角色"] == "CLIENT":
          ##这个任务明显就是多轮对话来判断对应意图表达
          _data_dict = {
            "instruction": system_prompt,
            "input":data["record_dt_info"],
            "output":data["label"]
          }
          data_dict['instruction_data'] =_data_dict
        else:
          data_dict["instruction_data"] = ""
        data_dict_list.append(data_dict)
    save_data = "data/example_test/test_intent.jsonl"
    with open(save_data,"w",encoding="utf-8") as file:
      file.write(json.dumps(data_dict_list,ensure_ascii=False,indent=4))
      
    
def read_dataset_file_to_intent_dataset(data_file_path):
    '''
    需要抽取client字段中 所有record_dt_info以及对应label 
    '''
    data = pd.read_csv(data_file_path)
    data['label_number'],total_label=pd.factorize(data["label"])
    loguru.logger.info(f"data info: {data.head()}")
    return data,total_label
  
def get_bot_last_convasation():
    '''
    bot最后一轮的
    '''
    pass
  
def augument_asr_convasation(data):
    '''
    需要对asr部分进行数据增强与澄清
    '''
    pass
  
  
# 地址（address）: **省**市**区**街**号，**路，**街道，**村等（如单独出现也标记）。地址是标记尽量完全的, 标记到最细。
# 书名（book）: 小说，杂志，习题集，教科书，教辅，地图册，食谱，书店里能买到的一类书籍，包含电子书。
# 公司（company）: **公司，**集团，**银行（央行，中国人民银行除外，二者属于政府机构）, 如：新东方，包含新华网/中国军网等。
# 游戏（game）: 常见的游戏，注意有一些从小说，电视剧改编的游戏，要分析具体场景到底是不是游戏。
# 政府（government）: 包括中央行政机关和地方行政机关两级。 中央行政机关有国务院、国务院组成部门（包括各部、委员会、中国人民银行和审计署）、国务院直属机构（如海关、税务、工商、环保总局等），军队等。
# 电影（movie）: 电影，也包括拍的一些在电影院上映的纪录片，如果是根据书名改编成电影，要根据场景上下文着重区分下是电影名字还是书名。
# 姓名（name）: 一般指人名，也包括小说里面的人物，宋江，武松，郭靖，小说里面的人物绰号：及时雨，花和尚，著名人物的别称，通过这个别称能对应到某个具体人物。
# 组织机构（organization）: 篮球队，足球队，乐团，社团等，另外包含小说里面的帮派如：少林寺，丐帮，铁掌帮，武当，峨眉等。
# 职位（position）: 古时候的职称：巡抚，知州，国师等。现代的总经理，记者，总裁，艺术家，收藏家等。
# 景点（scene）: 常见旅游景点如：长沙公园，深圳动物园，海洋馆，植物园，黄河，长江等。
  
  
  
def datasets_to_sft():
  # label_list = ["address","book","company","game","government","movie","name","organization","position","scene"]
  raw_dataset_path = "data/data_ner_example.csv"
  data_dict = pd.read_csv(raw_dataset_path)
  for index,row in data_dict.iterrows():
      messages = {
        "instruction":NER_TASK_LABEL_PROMPT
        
        
      }
    
    
def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        entity_names_list = set()
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input_text = data["text"]
            entities = data["entities"]
            # match_names = ["地点", "人名", "地理实体", "组织"]
            match_names = ['银行名称', '银行', '形容词', '产品名称', '产品', '金融产品', '金融名词', '银行产品']
            
            entity_sentence = ""
            
            for entity in entities:
                entity_json = dict(entity)
                entity_text = entity_json["entity_text"]
                entity_names = entity_json["entity_names"]
                for name in entity_names:
                    entity_names_list.add(name)
                    if name in match_names:
                        entity_label = name
                        break
                
                entity_sentence += f"""{{"entity_text": "{entity_text}", "entity_label": "{entity_label}"}}"""
            
            if entity_sentence == "":
                entity_sentence = "没有找到任何实体"
            
            message = {
                "instruction": """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """,
                "input": f"文本:{input_text}",
                "output": entity_sentence,
            }
            
            messages.append(message)
    loguru.logger.info(f"name list {entity_names_list}")
    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
      
    

if __name__ == "__main__":
    loguru.logger.info("race hub data preprocss start")
    # data_file_path = "data/data_example.csv"
    # # read_dataset_file_to_intent_dataset(data_file_path=data_file_path)
    # dataset_jsonl_transfer("data/bank.jsonl","data/bank_sft.jsonl")
    data_example_process()
    
    
    

        