# Document

## 1. 关于模型
这里使用的是XGBoost(原文查看[这里](http://arxiv.org/pdf/1603.02754v1.pdf))分类模型，这是一种对传统的GBDT算法的改进，
包括对损失函数，正则化，切分点查找等方面的优化。具体原理可以查看[这篇文章](https://blog.csdn.net/sb19931201/article/details/52557382)

UPDATE：经过评估后发现lightGBM能够达到更高的精度和更快的训练速度，关于LightGBM的原理可以查看[这里](https://www.imooc.com/article/39452)
，主要利用直方图决策树算法和带深度限制的leaf-wise叶子生长策略。

## 2. 代码解释
1）`preprocess.py`为数据预处理模块，主要包含以下3种处理方式：
1. 将非数值型数据（性别，地区，呼叫时间等）转换成onehot编码，主要为`preprocess_categorical_value`函数，其中对于缺失数据标记为**unknown**
2. 将数值型数据（年龄，税金等）进行一定形式的转化或保留原值输入，主要为`preprocess_numerical_value`函数，其中缺失数据用该列数据的**均值**替换。（函数中的normalize表示时候将数据进行mean-std均一化，use_log表示是否用log进行转换）
3. 根据该项数据是否存在（比如破产经历，是否为上市公司等）将原始数据转换成0/1

对输入模型的数据的具体预处理操作为：<br>
- "charger_id": 按'others', 'y-miyajima', 'oosaki', 'w-matsumoto', 'sakazaki', 'shimako', 'm-furukawa', 'a-ichikawa',
                 't-yamaguchi', 's-kinoshita', 'y-ozaki', 'miyake', 'a-go', 'omura', 'hoshina', 'y-endo',
                 'mik-sato', 'soma', 'h-oono', 'arakaki', 's-ootsu', 'saida'的顺序进行onehot编码
- "week":按'unknown', '土曜日', '日曜日', '月曜日', '木曜日', '水曜日', '火曜日', '金曜日'的顺序进行onehot编码，其中缺失数据为记为unknown
- "call_time": 取小时为单位（忽略分钟）将呼叫时间按1-24的书序进行onehot编码
- "list_type"：按'源泉', '管S'的顺序进行onehot编码
- "re_call_date"：根据时候有重新呼叫将有重新呼叫记为1，没有则记为0，然后进行onnehot编码
- "address"：以县为最小单位，按'unknown', 'others', '宮城県', '東京都', '愛知県', '大阪府', '埼玉県', '神奈川県', '千葉県'的顺序进行onehot编码，其中缺失数据记为unknown，除所给县以外的县记为others
- "kabushiki_code"：根据是否上市（是否有股票代码）进行onehot编码。
- "establishment"：根据所给设立时间计算公司的成立年数，缺失数值用所有有数据公司的平均年限代替
- "shihonkin"：缺失数值用所有有数据公司的平均设立资金代替
- "employee_num"：缺失数据用所有有数据公司的员工数代替
- "kojokazu"：有工厂的记为1，没有工厂的记为0，缺失的记为2，然后进行onehot编码
- "jigyoshokazu"：有事务所的记为1，没有的记为0，缺失的记为2，然后进行onehot编码
- "industry_code": 按'unknown', 'others', "721", "621", "631", "6821", "722", "6911", "6921", "641", "6812", "6941", "5229",
                      "831", "661", "9299", "5419", "651", "9121", "782", "5019", "6811", "5599", "6099",
                      "795", "5432", "841", "8821", "5319", "833", "5329", "4711"的顺序进行onehot编码，缺失记为unknown，除此以外的编号记为others
- "zenkikessan_uriagedaka"：缺失数据用所有有数据的平均前期营收代替
- "zenkikessan_riekikin"：缺失数据用所有有效数据的平均前期利润代替
- "tokikessan_uriagedaka"：缺失数据用所有有效数据的平均当期营收代替
- "tokikessan_riekikin"：缺失数据用所有有效数据的平均当期利润代替
- "tokiuriage_shinchoritsu"：缺失数据用所有有效数据的平均当期营收增长率代替
- "zenkiuriage_shinchoritsu"：缺失数据用所有有效数据的平均前期营收增长率代替
- "tokirieki_shinchoritsu"：缺失数据用所有有效数据的平均当期利润增长率代替
- "zenkirieki_shinchoritsu"：缺失数据用所有有效数据的平均前期利润增长率代替
- "birthday"：首先将生日转换成年龄，缺失数据用平均年龄代替
- "danjokubun"：将男性记为1，女性记为1，未知或数据缺失记为2
- "eto_meisho"：按12星座的顺序进行onehot编码，未知记为unknown
- "tosankeireki"：有破产经历记为1，没有或未知记为0
- "race_area"：按'unknown', '中国', '九州', '北海道', '北関東', '北陸', '四国', '東北', '東海', '甲信越', '関西', '首都圏'进行onehot编码，缺失数据记为unknown

    
2）`train.py`主要包含模型训练(`train`函数)和训练数据准备模块(`prepare_train_input`函数)，其中：
1. `run`函数模块包括xgboost分类模型的初始化和定义。其中`xgb_params`为超参数，超参数通过网格搜索进行选择。
2. 最后选择使用的数据维度通过添加和移除对比前后的精度差异来决定是否保留，最终筛选的得到的参数为：
```
"week", "call_time", "list_type", "re_call_date", "address",
"kabushiki_code", "establishment", "kojokazu", "shihonkin", "employee_num",
"tokikessan_riekikin", "tokikessan_uriagedaka", "tokirieki_shinchoritsu",
"race_area, "charger_id"
```
        
3) `predict.py`包含模型进行预测函数

## 3. 使用说明
1. 模型训练
```bash
python3 train.py --input_file=train_call_history.csv --model_name=cls.model
```
2. 模型预测
```bash
python3 predict.py --input_file=test_call_history.csv --model_name=cls.model --output_file=test_pred.txt
```
