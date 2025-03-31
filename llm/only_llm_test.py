# 定义日志文件路径
log_file_path = "cora_llama_only_test.log"

# 读取日志文件
with open(log_file_path, "r") as file:
    lines = file.readlines()

# 初始化计数器
correct = 0
total = 0

# 逐行解析日志
for line in lines:
    line = line.strip()  # 去掉空格和换行符
    if not line or '|' not in line:
        continue  # 跳过空行
    p = line.split(":")
    if not p[0].isdigit() : continue
    idx = int(p[0])
    if idx < 2216 : continue
    # 提取预测标签和真实标签
    parts = line.split(" | ")
    predict_label = parts[0].split(":")[-1].strip()
    true_label = parts[1].split(":")[-1].strip()
    
    # 比较标签是否一致
    if predict_label == true_label:
        correct += 1
    total += 1

# 计算准确率
accuracy = correct / total if total > 0 else 0

# 输出结果
print(f"正确预测数: {correct}")
print(f"总样本数: {total}")
print(f"准确率: {accuracy:.2%}")
