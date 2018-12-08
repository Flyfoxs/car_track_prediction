# 主要思路
1) 对每一个out_id, 单独学习,单独训练
2) 对每一个out_id, 根据之前的历史地址,确定阀值动态的对目的地地址进行归类. 如果out_id有太多历史目的地,可以适当增加阀值



# 运行步骤
1) 运行下面的程序,把训练和测试数据分为5块

python ./code_felix/split/group.py

2) 运行下面的程序对上面的5块分割分别进行训练

python ./code_felix/car/val_rf.py

3) 合并上一步的文件,生成可以提交的文件(里面的文件目录列表需要替换为真正的文件路径)

python ./code_felix/ensemble/concat.py