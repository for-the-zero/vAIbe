# TASK第二部分

根据PLAN.md和其它项目内的文件进行修改，不需要读取.cache和checkpoints和hfspace和__pycache__文件夹

1. 当前这个项目有一个很严重的问题，就是读取数据库没有使用utf-8编码，请修复
2. 我不要tatoeba.tsv数据集用于训练了，只需要cveto
3. 我希望这个扩散过程的噪声是一步一步渐变来的，慢慢在两个语言之间转换，不希望有突变的情况，当前情况是一开始就突变为一段看不出前后关联的同语言噪声，然后再进行到一般的时候突然变为另一个语言的噪声，找后面步骤细微的修改过后择偶为另一个语言的输出即：

```output
输入: hello
扩散过程:
前向扩散: en → 噪声空间
Step  980 → dine hearing skiing ge ures
Step  960 → pri hearing skiing ge ures
Step  940 → pri hearing ited ge ures
Step  920 → pri hearing ited ge ures
Step  900 → pri hearing ited ge ures
Step  880 → pri hearing ited ge ures
Step  860 → pri compared ited ge bi
Step  840 → pri compared ited ge bi
Step  820 → pri compared ited ge bi
Step  800 → pri compared ited ex states
Step  780 → pri compared ited ex states
Step  760 → busy compared ited ex states
Step  740 → busy compared ited ex states
Step  720 → leave compared ited ex states
Step  700 → leave compared ited ex states
Step  680 → cape swar ited co states
Step  660 → cape swar ited co states
Step  640 → cape swar ited co states
Step  620 → cape swar ited co states
Step  600 → cape swar ited co states
Step  580 → cape swar ited co states
Step  560 → cape swar ited co states
Step  540 → cape swar ited co sed
Step  520 → cape swar ited co sed
Step  500 → cape swar ited co sed
Step  480 → 话增位话好
Step  460 → 话增位话好
Step  440 → 话增位话好
Step  420 → 话增位话好
Step  400 → 话增位话好
Step  380 → 话增位话好
Step  360 → 话增位话好
Step  340 → 话增位话好
Step  320 → 话增位话好
Step  300 → 话增话话好
Step  280 → 话增话话好
Step  260 → 话增话话好
Step  240 → 话增话话好
Step  220 → 话增话话好
Step  200 → 话增话话好
Step  180 → 话增话话好
Step  160 → 话增话话好
Step  140 → 话增话话好
Step  120 → 话增话话该
Step  100 → 话增话话该
Step   80 → 话增话话该
Step   60 → 话增话话该
Step   40 → 话增话话该
Step   20 → 话增话话该
输出: 话增话话该
```

而我希望的是：

```output
输入: hello
扩散过程:
前向扩散: en → 噪声空间
Step  980 → hello
Step  850 → helio
Step  600 → 他lio
Step  500 → 他lio
Step  250 → 你行
Step  20 → 你好
输出: 你好
```

4. 要给出训练命令给我再新的终端执行，而不是自己完成完整训练
5. 尝试再训练的同时后台另一个线程tokenize，而不是同步处理，tokenize的缓存上线为50，缓存超过50暂停，直到缓存中的那个被用于训练了，再继续
6. 添加训练中途保存权重的功能，不打断当前训练进度，支持恢复，且支持恢复到特定的数据集编号
7. 沿着思路改进模型，使其效果更好