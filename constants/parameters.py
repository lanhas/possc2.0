

#--------------------------------------------- 冶炼参数设置 ------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

# 所有的输入属性
input_factorsAll = ['ingredient_C', 'ingredient_Si', 'ingredient_Mn', 'ingredient_P', 'ingredient_S', 'feLiquid_temp',
                    'fe_caliber', 'feLiquid_enclose', 'feScrapped_enclose', 'feLqCons_enclose', 'feRawCons_enclose', 'sum_enclose',
                    'steelLiquid', 'take_probality', 'oxygenSupply_time', 'oxygen_consume', 'stove_pull', 'lime_append', 'limestone_append',
                    'dolomite_append', 'mineral_append', 'qingshao_append', 'Mg_append', 'steelLiq_pullTemp1', 'steelLiq_addTime', 'steelLiq_outTemp',
                    'nitrogen_consume', 'nitrogen_time', 'slag_modifier', 'alloy_SiFe', 'alloy_SiMn', 'alloy_VN', 'alloy_SiAlFe', 'alloy_MnFe_hC',
                    'alloy_SiAlCa', 'alloy_MnFe_mC', 'alloy_TiFe', 'alloy_NbFe', 'alloy_Al', 'alloy_carburant']

# 所有的输出属性
output_factorsAll = ['terminal_C', 'terminal_S', 'terminal_P', 'terminal_Mn', 'steelLiq_pullTemp1', 'product_C', 'product_Si', 
                    'product_Mn', 'product_P', 'product_S', 'product_PaddS', 'product_Als']

# 测试时的输入属性
input_factorsTest = ['ingredient_C', 'ingredient_Si', 'ingredient_Mn', 'ingredient_P', 'ingredient_S', 'feLiquid_temp',
                    'feLiquid_enclose', 'feScrapped_enclose', 'feLqCons_enclose', 'feRawCons_enclose', 
                    'steelLiquid', 'oxygenSupply_time', 'oxygen_consume', 'lime_append', 'limestone_append',
                    'dolomite_append', 'mineral_append', 'qingshao_append', 'steelLiq_pullTemp1', 'nitrogen_time']

# 测试时的输出属性
output_factorsTest = ['terminal_C', 'terminal_S', 'terminal_P', 'terminal_Mn']

# 线性回归时的输出属性 
output_factorsRegression = ['terminal_C']

# 处理后表的中文属性
factors_zh = ['炉次', '计划钢种','C碳1', 'Si硅1', 'Mn锰1', 'P磷1', 'S硫1', '铁水温度（℃）', '出钢口龄', '铁水', 
                    '废钢', '铁水消耗', '钢铁料消耗', '合计', '钢水量', '收得率', '供氧时长', '耗氧量', '倒炉次数', 
                    '石灰', '石灰石', '生白云石', '矿石', '轻烧', '镁球', '一倒温度', '补吹时间', '出钢温度', '耗氮量', 
                    '溅渣用时', '调渣剂', '硅铁', '硅锰', '钒氮合金', ' 钢砂铝', '高碳锰铁', '硅铝钙', '中碳锰铁', 
                    '钛铁', '鈮铁', '铝粉', '增碳剂', 'C碳2', 'S硫2', 'P磷2', 'Mn锰2', 'C碳3', 'Si硅3', 'Mn锰3', 
                    'P磷3', 'S硫3', 'P+S之和', 'Als酸溶铝']
# 处理后表的英文属性
factors_en = ['stove_no', 'steel_type', 'ingredient_C', 'ingredient_Si', 'ingredient_Mn', 'ingredient_P', 
                    'ingredient_S', 'feLiquid_temp','fe_caliber', 'feLiquid_enclose', 'feScrapped_enclose',
                    'feLqCons_enclose', 'feRawCons_enclose', 'sum_enclose', 'steelLiquid', 'take_probality', 
                    'oxygenSupply_time', 'oxygen_consume', 'stove_pull', 'lime_append', 'limestone_append',
                    'dolomite_append', 'mineral_append', 'qingshao_append', 'Mg_append',
                    'steelLiq_pullTemp1', 'steelLiq_addTime', 'steelLiq_outTemp',
                    'nitrogen_consume', 'nitrogen_time', 'slag_modifier', 'alloy_SiFe',
                    'alloy_SiMn', 'alloy_VN', 'alloy_SiAlFe', 'alloy_MnFe_hC',
                    'alloy_SiAlCa', 'alloy_MnFe_mC', 'alloy_TiFe', 'alloy_NbFe', 'alloy_Al',
                    'alloy_carburant', 'terminal_C', 'terminal_S', 'terminal_P', 'terminal_Mn',
                    'product_C', 'product_Si', 'product_Mn', 'product_P', 'product_S',
                    'product_PaddS', 'product_Als']

# 时间类型属性（用于数据清洗中进行格式转换）
factors_time = ['oxygenSupply_time', 'nitrogen_time']

# 百分数类型属性（用于数据清洗中进行格式转换）
factors_percent = ['take_probality']

# 符号类型属性（用于数据清洗中进行格式转换）
factors_symbol = ['ingredient_C', 'ingredient_Si', 'ingredient_Mn', 'ingredient_P', 
                    'ingredient_S', 'feLiquid_temp','fe_caliber', 'feLiquid_enclose', 'feScrapped_enclose',
                    'feLqCons_enclose', 'feRawCons_enclose', 'sum_enclose', 'steelLiquid', 
                    'oxygen_consume', 'stove_pull', 'lime_append', 'limestone_append',
                    'dolomite_append', 'mineral_append', 'qingshao_append', 'Mg_append',
                    'steelLiq_pullTemp1', 'steelLiq_addTime', 'steelLiq_outTemp',
                    'nitrogen_consume', 'slag_modifier', 'alloy_SiFe',
                    'alloy_SiMn', 'alloy_VN', 'alloy_SiAlFe', 'alloy_MnFe_hC',
                    'alloy_SiAlCa', 'alloy_MnFe_mC', 'alloy_TiFe', 'alloy_NbFe', 'alloy_Al',
                    'alloy_carburant', 'terminal_C', 'terminal_S', 'terminal_P', 'terminal_Mn',
                    'product_C', 'product_Si', 'product_Mn', 'product_P', 'product_S',
                    'product_PaddS', 'product_Als']
# 原始表的原生属性（未去除无用字段）
index_builk_ori = ['生产日期', '班次', '炉次', '炉龄', '计划钢种', '氧枪枪龄（东枪）', '氧枪枪龄（西枪）', '氧枪模式',
                    '铁水成分C', '铁水成分Si', '铁水成分Mn', '铁水成分P', '铁水成分S', '铁水温度（℃）', '出钢口龄',
                    '铁水装入量', '计量废钢装入量', '自产废钢装入量', '铁水消耗量', '钢铁料消耗量', '合计', '钢水量', '钢水收得率',
                    '供氧时长', '耗氧量', '倒炉次数 ', '石灰加入量（渣料）', '石灰石加入量（渣料）', '生白云石加入量（渣料）',
                    '矿石加入量（渣料）', '轻烧加入量（渣料）', '镁球加入量（渣料）','钛铁加入量（渣料）', '渣料加入时间', '一倒钢水温度', '补吹时间',
                    '出钢温度', '耗氮量N㎡', '溅渣用时（分、秒）', '调渣剂kg', '硅铁加入量（合金）', '硅锰加入量（合金）',
                    '钒氮合金加入量（合金）', '硅锰合金球加入量（合金）', '渣洗料加入量（合金）', '钢砂铝加入量（合金）',
                    '高碳锰铁加入量（合金）', '硅铝钙加入量（合金）', '中碳锰铁加入量（合金）', '钛铁加入量（合金）', '铌铁加入量（合金）',
                    '铝粉加入量（合金）', '增碳剂加入量（合金）', '合金加入时间', '终点成分C', '终点成分S', '终点成分P',
                    '终点成分Mn', '刚包内钢水氧', '钢水去向', '成品成分C', '成品成分Si', '成品成分Mn', '成品成分P',
                    '成品成分S', '成品成分P+S', '成品成分Als酸溶铝', '成品成分钛', '成品成分铌', '成品成分其他', '挡渣标',
                    '兑铁次数', '备注']
# 原始表的中文属性
index_builk_zn = ['生产日期','炉次','班次','炉号','炉龄', '计划钢种', '氧枪枪龄（东枪）', '氧枪枪龄（西枪）', '氧枪模式',
                    '铁水成分C', '铁水成分Si', '铁水成分Mn', '铁水成分P', '铁水成分S', '铁水温度（℃）',
                    '出钢口龄','铁水装入量', '计量废钢装入量', '自产废钢装入量','钢水量',  '供氧时长', '耗氧量', '倒炉次数 ', 
                    '石灰加入量（渣料）', '石灰石加入量（渣料）', '生白云石加入量（渣料）','矿石加入量（渣料）', '轻烧加入量（渣料）', '镁球加入量（渣料）', '钛铁加入量（渣料）', '渣料加入时间', 
                    '一倒钢水温度', '补吹时间','出钢温度', '耗氮量N㎡', '溅渣用时（分、秒）', '调渣剂kg', 
                    '硅铁加入量（合金）', '硅锰加入量（合金）','钒氮合金加入量（合金）', '硅锰合金球加入量（合金）', '渣洗料加入量（合金）', '钢砂铝加入量（合金）',
                    '高碳锰铁加入量（合金）', '硅铝钙加入量（合金）', '中碳锰铁加入量（合金）', '钛铁加入量（合金）', '铌铁加入量（合金）','铝粉加入量（合金）', '增碳剂加入量（合金）', '合金加入时间', 
                    '终点成分C', '终点成分S', '终点成分P','终点成分Mn', '钢水去向', 
                    '成品成分C', '成品成分Si', '成品成分Mn', '成品成分P','成品成分S','成品成分Als酸溶铝', '成品成分钛', '成品成分铌', 
                    '兑铁次数', '备注']  #去除无用字段后的字段列表
# 原始表的英文属性
index_builk_eng = ['produce_date', 'furnace_num', 'class_no','furnace_no','furnace_age','steel_name','oxygun1','oxygun2','oxymode',
                    'iron_c','iron_si','iron_mn','iron_p','iron_s','iron_temperature',
                    'outlet_age','iron_weight','resteel_weight1','resteel_weight2','steel_weight','pro_oxytime1','pro_oxyval','turn_num',
                    '石灰加入量（渣料）', '石灰石加入量（渣料）', '生白云石加入量（渣料）','矿石加入量（渣料）', '轻烧加入量（渣料）', '镁球加入量（渣料）', '钛铁加入量（渣料）', '渣料加入时间', 
                    'turn_temperature1','pro_oxytime2','turn_temperature2','pro_nval','slag_time','slag_weight',
                    '硅铁加入量（合金）', '硅锰加入量（合金）','钒氮合金加入量（合金）', '硅锰合金球加入量（合金）', '渣洗料加入量（合金）', '钢砂铝加入量（合金）',
                    '高碳锰铁加入量（合金）', '硅铝钙加入量（合金）', '中碳锰铁加入量（合金）', '钛铁加入量（合金）', '铌铁加入量（合金）','铝粉加入量（合金）', '增碳剂加入量（合金）', '合金加入时间', 
                    'turn_c','turn_s','turn_p','turn_mn','steel_dir',
                    'steel_c','steel_si','steel_mn','steel_p','steel_s','steel_als','steel_ti','steel_nb',
                    'steel_fe','steel_remark']  #所有可能会使用的字段列表

# 产品属性
index_product_parameter = ['produce_date', 'furnace_num', 'class_no','furnace_no','furnace_age','steel_name','oxygun1','oxygun2','oxymode',
                    'iron_c','iron_si','iron_mn','iron_p','iron_s','iron_temperature',
                    'outlet_age','iron_weight','resteel_weight1','resteel_weight2','steel_weight','pro_oxytime1','pro_oxyval','turn_num','turn_temperature1','pro_oxytime2',
                    'turn_temperature1','pro_oxytime2','turn_temperature2','pro_nval','slag_time','slag_weight',
                    'turn_c','turn_s','turn_p','turn_mn','steel_dir',
                    'steel_fe','steel_remark']

index_product_accessory = ['produce_date','class_no','furnace_num','accessory_name','addweight','addtime']   #product_accessory中的字段名
index_product_alloy =  ['produce_date','class_no','furnace_num','alloy_name','addweight','addtime']  #product_alloy中的字段名
index_product_steel = ['steel_name','class_no','produce_date','furnace_num',
                    'steel_c','steel_si','steel_mn','steel_p','steel_s','steel_als','steel_ti','steel_nb']  #product_steel中的字段名


